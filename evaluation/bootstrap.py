"""Paired bootstrap uncertainty for fixed genetic scores.

All methods share each resample. Binary resampling preserves the numbers of
cases and controls. Only the evaluation regressions are refitted, not the PRS.
"""

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd

from blocprs.metrics import IncrementalR2, NagelkerkeR2, NonEstimableError


@dataclass(frozen=True)
class BootstrapResult:
    """Observed metrics, paired replicates, and per-method uncertainty."""

    methods: tuple[str, ...]
    observed: np.ndarray
    replicates: np.ndarray
    method_summary: pd.DataFrame


# Reuse one covariate-only fit across all methods in this sample.
def _metric_values(y, covariates, scores, methods, metric_type, label):
    try:
        metric = metric_type(y, covariates)
    except (NonEstimableError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise NonEstimableError(f"{label}: covariate-only fit failed: {exc}") from exc

    values = np.empty(len(methods), dtype=np.float64)
    for column, method in enumerate(methods):
        try:
            values[column] = metric.score(scores[:, column])
        except (NonEstimableError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise NonEstimableError(f"{label}: method {method!r} failed: {exc}") from exc

    return values


# Resample participants, retaining the same participant draw for every method.
def bootstrap_metrics(y, covariates, scores, *, family, n_boot=10000, seed=20260731):
    """Return paired uncertainty from aligned phenotype, covariates, and scores.

    Scores may be a DataFrame or a mapping of method names to vectors. Rows must
    already be aligned. Any non-estimable replicate stops the calculation;
    replicates are never discarded or replaced. Intervals are 95% percentiles.
    """

    if family not in ("gaussian", "binomial"):
        raise ValueError("family must be gaussian or binomial.")

    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError("n_boot must be an integer of at least 2.")

    y = np.asarray(y, dtype=np.float64)
    covariates = None if covariates is None else np.asarray(covariates, dtype=np.float64)
    table = pd.DataFrame(scores)
    methods = tuple(table.columns)

    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Provide at least one score with unique method names.")

    if any(not isinstance(name, str) or not name.strip() for name in methods):
        raise ValueError("Method names must be nonempty strings.")

    values = table.to_numpy(dtype=np.float64)

    if y.ndim != 1 or values.shape[0] != len(y) or not np.isfinite(values).all():
        raise ValueError("Every method needs a finite score for every participant.")

    binary = family == "binomial"
    metric_type = NagelkerkeR2 if binary else IncrementalR2
    observed = _metric_values(y, covariates, values, methods, metric_type, "Observed sample")
    replicates = np.empty((n_boot, len(methods)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    participants = np.arange(len(y))

    # For binary phenotypes, resample controls and cases separately.
    strata = [np.flatnonzero(y == value) for value in (0.0, 1.0)] if binary else [participants]
    probabilities = [np.full(len(rows), 1.0 / len(rows)) for rows in strata]

    for replicate in range(n_boot):
        counts = np.zeros(len(y), dtype=np.int64)
        for rows, probability in zip(strata, probabilities):
            counts[rows] = rng.multinomial(len(rows), probability)

        # Repeated rows let us use exactly the same metric implementation as CV.
        selected = np.repeat(participants, counts)
        selected_covariates = None if covariates is None else covariates[selected]
        replicates[replicate] = _metric_values(y[selected], selected_covariates, values[selected],
                                               methods, metric_type, f"Bootstrap replicate {replicate + 1}")

        # Release reference cycles in the logistic regression results.
        if binary:
            gc.collect()

    intervals = np.quantile(replicates, [0.025, 0.975], axis=0)
    summary = pd.DataFrame({"method": methods, "observed": observed,
                            "bootstrap_se": replicates.std(axis=0, ddof=1),
                            "ci_lower": intervals[0], "ci_upper": intervals[1], "n_boot": n_boot})
    return BootstrapResult(methods, observed, replicates, summary)


# Summarize paired metric differences, each method minus the named reference.
def paired_comparisons(result, reference):
    if reference not in result.methods:
        raise ValueError(f"Reference method {reference!r} is absent from the scores.")

    reference_column = result.methods.index(reference)
    rows = []
    for column, method in enumerate(result.methods):
        if column == reference_column:
            continue

        differences = result.replicates[:, column] - result.replicates[:, reference_column]
        lower, upper = np.quantile(differences, [0.025, 0.975])
        rows.append({"method": method, "reference": reference,
                     "difference": result.observed[column] - result.observed[reference_column],
                     "bootstrap_se": differences.std(ddof=1), "ci_lower": lower, "ci_upper": upper,
                     "n_boot": len(differences)})

    return pd.DataFrame(rows, columns=["method", "reference", "difference", "bootstrap_se",
                                       "ci_lower", "ci_upper", "n_boot"])
