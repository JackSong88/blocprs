"""Choose the regional penalty by cross-validation within the training data."""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .data import Design
from .metrics import IncrementalR2, NagelkerkeR2
from .solvers import continuous_statistics, fit


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


# Select lambda by mean validation score without refitting the final model here.
# Keep the supplied training means and SDs fixed throughout CV.
def select_lambda(data, scaler, family, lambda_grid, *, cv_folds=5, cv_max_samples=50000,
                  random_state=42, block_size=8192):
    if family not in ("gaussian", "binomial"):
        raise ValueError("family must be 'gaussian' or 'binomial'.")
    cv_folds = _positive_integer(cv_folds, "cv_folds", minimum=2)
    block_size = _positive_integer(block_size, "block_size")
    if cv_max_samples is not None:
        cv_max_samples = _positive_integer(cv_max_samples, "cv_max_samples", minimum=cv_folds)

    candidates = np.asarray(lambda_grid, dtype=np.float64)
    if (candidates.ndim != 1
        or candidates.size == 0
        or not np.isfinite(candidates).all()
        or np.any(candidates < 0)
        or len(np.unique(candidates)) != len(candidates)):
        raise ValueError("lambda_grid must contain distinct finite nonnegative values.")
    if family == "binomial":

        # Try larger penalties first; reuse fitted coefficients within each fold.
        candidates = np.sort(candidates)[::-1]

    full = Design(data, scaler, block_size=block_size)
    y = np.asarray(full.outcome(), dtype=np.float64)
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
    folds = splitter.split(np.zeros((total_used, 1)), sampled_y)
    for fold_index, (train_rows, val_rows) in enumerate(folds):
        train = Design(data, scaler, rows=sampled[train_rows], block_size=block_size)
        validation = Design(data, scaler, rows=sampled[val_rows], block_size=block_size)
        metric_type = NagelkerkeR2 if family == "binomial" else IncrementalR2
        try:
            metric = metric_type(validation.outcome(), validation.covariates())
        except Exception as exc:
            raise ValueError(f"CV fold {fold_index + 1}: auxiliary null fit failed: {exc}") from exc

        statistics = continuous_statistics(train) if family == "gaussian" else None
        n_unpenalized = 1 + train.n_covariates + len(data.source_names)
        initial = None

        for entry in results:
            try:
                fitted = fit(train, family, entry["lambda"], n_unpenalized, initial=initial, statistics=statistics)
                score = float(metric.score(_genetic_predictions(validation, fitted.coef)))
            except Exception as exc:
                raise ValueError(f"CV fold {fold_index + 1}, lambda={entry['lambda']:g}: {exc}") from exc

            entry["fold_scores"].append(score)
            if family == "binomial":
                initial = fitted.coef

    for entry in results:
        entry["mean_score"] = float(np.mean(entry["fold_scores"]))

    # Ties retain the first evaluated candidate.
    selected = max(results, key=lambda entry: entry["mean_score"])
    return selected["lambda"], results
