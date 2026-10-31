"""AUROC and paired DeLong uncertainty for fixed genetic scores."""

import math

import numpy as np
import pandas as pd


# Compare cases with controls, giving tied scores half credit.
def delong_covariance(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 2 or scores.shape[0] != len(y):
        raise ValueError("Scores need one row per participant and one column per method.")

    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("AUROC requires binary outcomes and finite scores.")

    m, n = int(np.sum(y == 1)), int(np.sum(y == 0))

    if not m or not n:
        return np.full(scores.shape[1], np.nan), np.full((scores.shape[1], scores.shape[1]), np.nan)

    case_placements, control_placements = [], []
    for score in scores.T:
        cases, controls = score[y == 1], score[y == 0]
        sorted_cases, sorted_controls = np.sort(cases), np.sort(controls)

        # Insertion ranks count lower scores; averaging both sides gives ties half credit.
        # Sorting avoids constructing the full case-by-control comparison matrix.
        left = np.searchsorted(sorted_controls, cases, side="left")
        right = np.searchsorted(sorted_controls, cases, side="right")
        case_placements.append((left + right) / (2 * n))
        left = np.searchsorted(sorted_cases, controls, side="left")
        right = np.searchsorted(sorted_cases, controls, side="right")
        control_placements.append(1 - (left + right) / (2 * m))

    v, w = np.asarray(case_placements), np.asarray(control_placements)
    covariance = np.full((scores.shape[1], scores.shape[1]), np.nan)

    if min(m, n) >= 2:
        covariance = np.atleast_2d(np.cov(v, ddof=1)) / m + np.atleast_2d(np.cov(w, ddof=1)) / n

    return v.mean(axis=1), covariance


# Report raw-score AUROC without fitting covariates or reversing score directions.
def evaluate_auroc(y, scores, references=()):
    y = np.asarray(y)
    methods = list(scores)
    values, covariance = delong_covariance(y, scores.to_numpy(dtype=float))
    cases, controls = int(np.sum(y == 1)), int(np.sum(y == 0))
    eligible = min(cases, controls) >= 10
    rows, comparisons = [], []

    for j, method in enumerate(methods):
        reason = "" if np.isfinite(values[j]) else "AUROC requires both cases and controls."
        se = np.sqrt(max(0, covariance[j, j])) if eligible else np.nan
        rows.append(dict(method=method, metric="auroc", value=values[j], n=len(y),
                         cases=cases, controls=controls, status="NE" if reason else "OK",
                         reason=reason, se=se, uncertainty_method="delong",
                         uncertainty_status="OK" if eligible else "sparse_class"))

        for reference in references:
            k = methods.index(reference)

            if j == k:
                continue

            difference = values[j] - values[k]

            # Shared participants make the two AUROCs correlated, so retain their covariance.
            variance = covariance[j, j] + covariance[k, k] - 2 * covariance[j, k]
            se, p, status = np.nan, np.nan, "sparse_class"

            if eligible:

                if not np.isfinite(variance) or variance < -1e-12:
                    raise ValueError("Invalid paired DeLong variance.")

                if variance <= 1e-15:
                    status = "identical_auc_zero_variance" if abs(difference) <= 1e-12 else "degenerate_variance"
                    se, p = (0.0, 1.0) if abs(difference) <= 1e-12 else (np.nan, np.nan)
                else:
                    se = math.sqrt(variance)

                    # erfc gives the two-sided normal P value from the absolute z-statistic.
                    p, status = math.erfc(abs(difference / se) / math.sqrt(2)), "OK"

            comparisons.append(dict(method=method, reference=reference, metric="auroc",
                                    difference=difference, difference_se=se, p_value=p,
                                    n=len(y), cases=cases, controls=controls, status=status,
                                    uncertainty_method="delong"))

    return pd.DataFrame(rows), pd.DataFrame(comparisons)
