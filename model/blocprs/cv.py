"""Cross-validated selection of the regional regularization parameter.

Genetic coefficients are estimated on training folds and evaluated on
validation folds using covariate-adjusted R^2.
"""

from contextlib import ExitStack
import logging
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from .cache import cached_design, stage_cv_data, workspace
from .data import Design, Scaler
from .metrics import IncrementalR2, NagelkerkeR2
from .solvers import continuous_statistics, fit


logger = logging.getLogger(__name__)


# Draw a random CV subset, stratified by outcome when requested.
def _subsample_indices(y, total_used, rng, stratify=False, min_per_class=1):
    n_samples = len(y)
    all_idx = np.arange(n_samples)

    if total_used >= n_samples:
        return all_idx

    if not stratify:
        return rng.permutation(all_idx)[:total_used]

    classes, inverse = np.unique(y, return_inverse=True)

    if len(classes) < 2:
        raise ValueError("Binary CV requires both outcome classes.")

    class_indices = [np.flatnonzero(inverse == i) for i in range(len(classes))]
    counts = np.array([len(idx) for idx in class_indices], dtype=np.int64)

    # Allocate proportional class counts, bounded by the available rows.
    expected = total_used * counts.astype(np.float64) / float(n_samples)
    desired = np.floor(expected).astype(np.int64)
    min_counts = np.minimum(counts, max(int(min_per_class), 1))

    if int(min_counts.sum()) <= total_used:
        desired = np.maximum(desired, min_counts)

    desired = np.minimum(desired, counts)

    # Correct rounding to reach the exact sample size without exceeding class bounds.
    current = int(desired.sum())
    fractions = expected - np.floor(expected)

    if current < total_used:
        order = np.argsort(-fractions)
        while current < total_used:
            changed = False
            for index in order:
                if desired[index] < counts[index]:
                    desired[index] += 1
                    current += 1
                    changed = True

                    if current == total_used:
                        break

            if not changed:
                raise ValueError("Could not allocate the requested CV sample.")
    elif current > total_used:
        lower = min_counts if int(min_counts.sum()) <= total_used else np.zeros_like(desired)
        order = np.argsort(fractions)
        while current > total_used:
            changed = False
            for index in order:
                if desired[index] > lower[index]:
                    desired[index] -= 1
                    current -= 1
                    changed = True

                    if current == total_used:
                        break

            if not changed:
                raise ValueError("Could not allocate the requested CV sample.")

    sampled = np.concatenate([rng.permutation(indices)[:int(count)]
                              for indices, count in zip(class_indices, desired) if count > 0])
    rng.shuffle(sampled)
    return sampled[:total_used]


# Require an integer count that meets the specified minimum.
def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")

    return int(value)


# Predict genetics only, excluding the fitted intercept and covariates.
def _genetic_predictions(design, coef):
    first_genetic = 1 + design.n_covariates
    prediction = np.empty(design.n_samples, dtype=np.float64)
    offset = 0

    for X, _ in design.iter_blocks():
        stop = offset + len(X)
        prediction[offset:stop] = X[:, first_genetic:] @ coef[first_genetic:]
        offset = stop

    if offset != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError("Invalid validation genetic predictions.")

    return prediction


# Fit a complete penalty path within one fold; its scaler and warm starts stay local.
def _fit_fold(data, family, candidates, train_rows, val_rows, fold_index, block_size,
              cache_dir, log_level=logging.WARNING):
    logging.basicConfig(level=log_level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    started = perf_counter()
    with threadpool_limits(limits=1), ExitStack() as stack:
        return _fit_fold_path(data, family, candidates, train_rows, val_rows, fold_index,
                              block_size, cache_dir, stack, started)


# Cache this fold's transformed inputs only after fitting its own scaler.
def _fit_fold_path(data, family, candidates, train_rows, val_rows, fold_index,
                   block_size, cache_dir, stack, started):
    scaler = Scaler.fit(data, family, block_size, rows=train_rows)
    train = Design(data, scaler, rows=train_rows, block_size=block_size)
    validation = Design(data, scaler, rows=val_rows, block_size=block_size)

    # Refit only the evaluation regressions on validation rows, using a frozen PRS.
    metric_type = NagelkerkeR2 if family == "binomial" else IncrementalR2
    try:
        metric = metric_type(validation.outcome(), validation.covariates())
    except Exception as exc:
        raise ValueError(f"CV fold {fold_index + 1}: auxiliary null fit failed: {exc}") from exc

    validation = stack.enter_context(cached_design(validation, cache_dir, f"fold_{fold_index + 1}_validation"))

    if family == "binomial":
        train = stack.enter_context(cached_design(train, cache_dir, f"fold_{fold_index + 1}_train"))

    # Gaussian candidates reuse cross-products; binary candidates reuse the previous optimum.
    statistics = continuous_statistics(train) if family == "gaussian" else None
    n_unpenalized = 1 + train.n_covariates + len(data.source_names)
    initial = None
    scores = []
    for number, penalty in enumerate(candidates, 1):
        try:
            fitted = fit(train, family, penalty, n_unpenalized, initial=initial, statistics=statistics)
            score = float(metric.score(_genetic_predictions(validation, fitted.coef)))
        except Exception as exc:
            raise ValueError(f"CV fold {fold_index + 1}, lambda={penalty:g}: {exc}") from exc

        scores.append(score)
        if family == "binomial":
            initial = fitted.coef

        if number == 1 or number % 5 == 0 or number == len(candidates):
            logger.info("CV fold %s: penalty %s/%s, lambda=%g, score=%.8g, elapsed=%.1fs",
                        fold_index + 1, number, len(candidates), penalty, score, perf_counter() - started)

    return scores


# Select lambda by mean validation score without refitting the final model here.
# Learn means and SDs on each fold's training rows, then freeze for validation.
def select_lambda(data, family, lambda_grid, *, cv_folds=5, cv_max_samples=50000,
                  random_state=42, block_size=8192, cv_workers=1, cache_dir=None):
    if family not in ("gaussian", "binomial"):
        raise ValueError("family must be 'gaussian' or 'binomial'.")

    cv_folds = _positive_integer(cv_folds, "cv_folds", minimum=2)
    cv_workers = min(_positive_integer(cv_workers, "cv_workers"), cv_folds)
    block_size = _positive_integer(block_size, "block_size")

    if cv_max_samples is not None:
        cv_max_samples = _positive_integer(cv_max_samples, "cv_max_samples", minimum=cv_folds)

    candidates = np.asarray(lambda_grid, dtype=np.float64)

    if (candidates.ndim != 1 or candidates.size == 0
        or not np.isfinite(candidates).all() or np.any(candidates < 0)
        or len(np.unique(candidates)) != len(candidates)):
        raise ValueError("lambda_grid must contain distinct finite nonnegative values.")

    # Start binary fits at larger penalties, then reuse coefficients within each fold.
    if family == "binomial":
        candidates = np.sort(candidates)[::-1]

    y = np.asarray(data.y, dtype=np.float64)

    if y.shape != (data.n_samples,) or not np.isfinite(y).all():
        raise ValueError("CV requires a finite training outcome vector.")

    available = data.n_samples if cv_max_samples is None else min(data.n_samples, cv_max_samples)

    # Round down so each validation fold has the same number of rows.
    total_used = (available // cv_folds) * cv_folds

    if total_used < cv_folds:
        raise ValueError("Too few training participants for the requested CV folds.")

    sampled = _subsample_indices(y, total_used, np.random.RandomState(random_state),
                                stratify=family == "binomial", min_per_class=cv_folds)
    sampled_y = y[sampled]

    if family == "binomial":
        values, counts = np.unique(sampled_y, return_counts=True)

        if not np.array_equal(values, [0.0, 1.0]) or np.min(counts) < cv_folds:
            raise ValueError("Binary CV needs at least cv_folds sampled participants in each class.")

        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results = [{"lambda": float(value), "mean_score": None, "fold_scores": []} for value in candidates]
    folds = list(splitter.split(np.zeros((total_used, 1)), sampled_y))
    logger.info("CV: %s participants, %s folds, %s penalties, %s workers",
                total_used, cv_folds, len(candidates), cv_workers)

    # Stage raw sampled rows once; each worker still fits its own training-fold scaler.
    n_features = 1 + data.n_covariates + data.n_sources * (1 + data.n_windows)
    raw_bytes = total_used * data.n_windows * data.n_sources * data.window_scores.dtype.itemsize
    cached_rows = total_used if family == "binomial" else total_used // cv_folds
    cache_bytes = raw_bytes + cv_workers * cached_rows * (n_features + 1) * 8

    with workspace(cache_dir, cache_bytes) as directory:
        with stage_cv_data(data, sampled, directory, block_size) as staged:
            log_level = logger.getEffectiveLevel()

            if cv_workers == 1:
                fold_scores = []
                for number, (train_rows, val_rows) in enumerate(folds):
                    fold_scores.append(_fit_fold(staged, family, candidates, train_rows,
                                                 val_rows, number, block_size, directory, log_level))
                    logger.info("CV fold %s/%s complete", number + 1, cv_folds)
            else:

                # Workers reopen shared raw maps, but their transformed caches remain fold-specific.
                # One numerical thread per fold prevents nested CPU oversubscription.
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    with Parallel(n_jobs=cv_workers, batch_size=1, pre_dispatch=cv_workers,
                                  return_as="generator", max_nbytes="1M", mmap_mode="r") as pool:
                        completed = pool(delayed(_fit_fold)(staged, family, candidates, train_rows,
                                                            val_rows, number, block_size, directory, log_level)
                                         for number, (train_rows, val_rows) in enumerate(folds))
                        fold_scores = []
                        for scores in completed:
                            fold_scores.append(scores)
                            logger.info("CV fold %s/%s collected", len(fold_scores), cv_folds)

    # Ordered results preserve fold labels and the existing tie-breaking rule.
    for scores in fold_scores:
        for entry, score in zip(results, scores):
            entry["fold_scores"].append(score)

    for entry in results:
        entry["mean_score"] = float(np.mean(entry["fold_scores"]))

    # Ties retain the first evaluated candidate.
    selected = max(results, key=lambda entry: entry["mean_score"])
    logger.info("CV selected lambda=%g, mean score=%.8g", selected["lambda"], selected["mean_score"])
    return selected["lambda"], results
