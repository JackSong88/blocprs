"""Paired bootstrap uncertainty for fixed genetic scores.

All methods share each resample. Binary resampling preserves the numbers of
cases and controls. Only the evaluation regressions are refitted, not the PRS.
"""

import gc
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

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


# Frequency weights represent repeated participants without expanding the arrays.
def _weighted_values(y, covariates, scores, methods, family, counts, label):
    keep = counts > 0
    weights = counts[keep].astype(float)
    y, scores = y[keep], scores[keep]
    covariates = np.empty((len(y), 0)) if covariates is None else covariates[keep]
    design = np.column_stack((np.ones(len(y)), covariates))

    # The likelihood normalization uses the resampled size, not the number of unique rows.
    n = weights.sum()

    if family == "gaussian":
        sqrt_weights = np.sqrt(weights)
        centered = y - np.average(y, weights=weights)
        total_ss = float(weights @ (centered * centered))

        if not np.isfinite(total_ss) or total_ss <= 0:
            raise NonEstimableError(f"{label}: zero weighted phenotype variance.")

        def fit(matrix):
            coef = np.linalg.lstsq(matrix * sqrt_weights[:, None], y * sqrt_weights, rcond=None)[0]
            residual = y - matrix @ coef
            return float(weights @ (residual * residual))

        def difference(null_value, full_value):
            return (null_value - full_value) / total_ss

    else:

        def fit(matrix):

            # Weighted centering and scaling improve IRLS conditioning without adding a penalty.
            predictors = matrix[:, 1:]
            centered = predictors - np.average(predictors, axis=0, weights=weights)
            variance = np.average(centered * centered, axis=0, weights=weights)
            varying = variance > 0
            matrix = np.column_stack((np.ones(len(y)), centered[:, varying] / np.sqrt(variance[varying])))
            with warnings.catch_warnings():
                warnings.simplefilter("error", PerfectSeparationWarning)
                result = sm.GLM(y, matrix, family=sm.families.Binomial(), freq_weights=weights).fit(
                    method="IRLS", maxiter=500, tol=1e-10)

            if not result.converged or not np.isfinite(result.llf) or not np.isfinite(result.params).all():
                raise NonEstimableError("Weighted logistic fit did not converge to a finite solution.")

            return float(result.llf)

        def difference(null_value, full_value):
            denominator = 1.0 - np.exp(2.0 * null_value / n)

            if not np.isfinite(denominator) or denominator <= 0:
                raise NonEstimableError("Weighted Nagelkerke normalization is not estimable.")

            return (1.0 - np.exp(-2.0 * (full_value - null_value) / n)) / denominator

    errors = (NonEstimableError, np.linalg.LinAlgError, FloatingPointError,
              PerfectSeparationError, PerfectSeparationWarning)
    try:
        null_value = fit(design)
    except errors as exc:
        raise NonEstimableError(f"{label}: covariate-only fit failed: {exc}") from exc

    values = np.empty(len(methods), dtype=float)
    for column, method in enumerate(methods):
        try:
            values[column] = difference(null_value, fit(np.column_stack((design, scores[:, column]))))

            if not np.isfinite(values[column]):
                raise NonEstimableError("Weighted metric is not finite.")

        except errors as exc:
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

        # Multinomial counts describe sampling with replacement within each outcome stratum.
        # Reuse the draw across methods to retain their paired dependence.
        counts = np.zeros(len(y), dtype=np.int64)
        for rows, probability in zip(strata, probabilities):
            counts[rows] = rng.multinomial(len(rows), probability)

        # Weighted regressions preserve the draw without making repeated copies of participants.
        replicates[replicate] = _weighted_values(y, covariates, values, methods, family, counts,
                                                  f"Bootstrap replicate {replicate + 1}")

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

        # Subtract within each shared resample; separate marginal intervals lose this pairing.
        differences = result.replicates[:, column] - result.replicates[:, reference_column]
        lower, upper = np.quantile(differences, [0.025, 0.975])
        rows.append({"method": method, "reference": reference,
                     "difference": result.observed[column] - result.observed[reference_column],
                     "bootstrap_se": differences.std(ddof=1), "ci_lower": lower, "ci_upper": upper,
                     "n_boot": len(differences)})

    return pd.DataFrame(rows, columns=["method", "reference", "difference", "bootstrap_se",
                                       "ci_lower", "ci_upper", "n_boot"])
