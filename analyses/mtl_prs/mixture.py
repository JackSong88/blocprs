"""Select a nonnegative mixture of raw source PRSs within MTL training data.

Each candidate sums to one. Regression coefficients are fitted on each training
fold and frozen on its validation fold. These held-out selection scores differ
from the auxiliary-regression R^2 used for final evaluation.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold


# Enumerate source weights in the original simplex-grid order.
def simplex_grid(n_sources, step=.05):
    if n_sources < 1 or not np.isfinite(step) or step <= 0:
        raise ValueError("Provide at least one source and a positive finite grid step")

    units = int(round(1 / step))

    if not np.isclose(1 / step, units):
        raise ValueError("The grid step must be a reciprocal integer, such as 0.05")

    if n_sources == 1:
        return np.ones((1, 1))

    rows = []
    for bars in combinations(range(units + n_sources - 1), n_sources - 1):
        boundaries = (-1, *bars, units + n_sources - 1)
        rows.append([boundaries[i + 1] - boundaries[i] - 1 for i in range(n_sources)])
    return np.asarray(rows, dtype=float) / units


# Fit unpenalized logistic regression with training-only numerical conditioning.
def _fit_logistic(y, design):
    means = design.mean(axis=0)
    sd = design.std(axis=0, ddof=0)
    varying = np.isfinite(sd) & (sd > 0)
    conditioned = design.copy()
    conditioned[:, varying] = (conditioned[:, varying] - means[varying]) / sd[varying]

    # Sum binary negative log-likelihood and its analytic gradient.
    def objective(coef):
        eta = conditioned @ coef
        loss = np.logaddexp(0, eta).sum() - y @ eta
        gradient = conditioned.T @ (expit(eta) - y)
        return float(loss), gradient

    fitted = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
                      options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8, "maxls": 50})

    if not fitted.success or not np.isfinite(fitted.x).all() or not np.isfinite(fitted.fun):
        raise RuntimeError(f"Mixture-CV logistic fit failed: {fitted.message}")

    return fitted.x, means, sd


# Evaluate the frozen logistic model using its training means and SDs.
def _log_likelihood(y, design, model):
    coef, means, sd = model
    varying = np.isfinite(sd) & (sd > 0)
    conditioned = design.copy()
    conditioned[:, varying] = (conditioned[:, varying] - means[varying]) / sd[varying]
    eta = conditioned @ coef
    return -float(np.logaddexp(0, eta).sum() - y @ eta)


# Evaluate all Gaussian mixtures from small source-by-source covariance matrices.
def _gaussian_scores(train_y, train_scores, train_c, valid_y, valid_scores, valid_c, grid):
    covariate_y_coef = np.linalg.lstsq(train_c, train_y, rcond=None)[0]
    covariate_score_coef = np.linalg.lstsq(train_c, train_scores, rcond=None)[0]
    train_y_residual = train_y - train_c @ covariate_y_coef
    train_scores_residual = train_scores - train_c @ covariate_score_coef
    valid_y_residual = valid_y - valid_c @ covariate_y_coef
    valid_scores_adjusted = valid_scores - valid_c @ covariate_score_coef

    train_cross = train_scores_residual.T @ train_y_residual
    train_gram = train_scores_residual.T @ train_scores_residual

    # Each row computes pi.T @ train_gram @ pi, the mixture's residual sum of squares.
    denominator = np.einsum("ij,jk,ik->i", grid, train_gram, grid, optimize=True)

    # A mixture with no residual training variation receives a zero PRS coefficient.
    score_coef = np.divide(grid @ train_cross, denominator, out=np.zeros(len(grid)), where=denominator > 0)
    valid_cross = valid_scores_adjusted.T @ valid_y_residual
    valid_gram = valid_scores_adjusted.T @ valid_scores_adjusted
    covariate_sse = float(valid_y_residual @ valid_y_residual)

    if covariate_sse <= 0:
        raise ValueError("Gaussian mixture CV requires positive covariate-only validation error")

    # Expand validation SSE using the training-fitted coefficients; do not refit here.
    full_sse = (covariate_sse - 2 * score_coef * (grid @ valid_cross)
                + score_coef**2 * np.einsum("ij,jk,ik->i", grid, valid_gram, grid, optimize=True))
    return 1 - full_sse / covariate_sse


# Evaluate each binary mixture with training-fitted, frozen logistic regressions.
def _binary_scores(train_y, train_scores, train_c, valid_y, valid_scores, valid_c, grid):
    null_model = _fit_logistic(train_y, train_c)
    null_ll = _log_likelihood(valid_y, valid_c, null_model)
    normalization = 1 - np.exp(2 * null_ll / len(valid_y))

    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Binary mixture CV has a nonestimable likelihood normalization")

    metrics = np.empty(len(grid))
    for index, pi in enumerate(grid):
        train_full = np.column_stack((train_c, train_scores @ pi))
        valid_full = np.column_stack((valid_c, valid_scores @ pi))
        full_model = _fit_logistic(train_y, train_full)
        full_ll = _log_likelihood(valid_y, valid_full, full_model)
        metrics[index] = (1 - np.exp(-2 * (full_ll - null_ll) / len(valid_y))) / normalization
    return metrics


# Select the highest mean held-out score, retaining every candidate and fold.
def select_mixture(y, scores, covariates, source_names, *, family="gaussian", folds=5, step=.05, seed=42):
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    covariates = np.empty((len(y), 0)) if covariates is None else np.asarray(covariates, dtype=float)
    source_names = tuple(map(str, source_names))

    if covariates.ndim == 1:
        covariates = covariates[:, None]

    if y.ndim != 1 or scores.ndim != 2 or covariates.ndim != 2:
        raise ValueError("Provide an outcome vector and participant-by-column score/covariate arrays")
    if scores.shape[0] != len(y) or covariates.shape[0] != len(y):
        raise ValueError("Outcomes, scores and covariates must contain the same participants")
    if len(source_names) != scores.shape[1] or len(set(source_names)) != len(source_names):
        raise ValueError("Provide one unique name per source score")
    if not all(np.isfinite(values).all() for values in (y, scores, covariates)):
        raise ValueError("Mixture training data must be finite")
    if family not in ("gaussian", "binomial") or not 2 <= folds <= len(y):
        raise ValueError("Choose gaussian or binomial and between 2 and N folds")

    if family == "binomial":
        classes, counts = np.unique(y, return_counts=True)

        if not np.array_equal(classes, [0, 1]) or counts.min() < folds:
            raise ValueError("Binary mixture CV requires 0/1 outcomes and at least one case/control per fold")

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        validation_folds = [valid for _, valid in splitter.split(np.zeros(len(y)), y)]
    else:
        permutation = np.random.RandomState(seed).permutation(len(y))
        validation_folds = np.array_split(permutation, folds)

    grid = simplex_grid(len(source_names), step)
    fold_metrics = np.empty((folds, len(grid)))
    design = np.column_stack((np.ones(len(y)), covariates))
    score_fold = _binary_scores if family == "binomial" else _gaussian_scores
    for index, valid in enumerate(validation_folds):
        training = np.ones(len(y), dtype=bool)
        training[valid] = False
        train_scores = scores[training]
        sd = train_scores.std(axis=0, ddof=1)

        # Source SDs detect degenerate inputs; the simplex always uses raw PRSs.
        if not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError(f"A source PRS has zero/nonfinite training SD in fold {index}")

        fold_metrics[index] = score_fold(y[training], train_scores, design[training],
                                         y[valid], scores[valid], design[valid], grid)

        if not np.isfinite(fold_metrics[index]).all():
            raise RuntimeError(f"A mixture candidate has a nonfinite metric in fold {index}")

    mean_metric = fold_metrics.mean(axis=0)
    best = int(np.argmax(mean_metric))
    candidates = pd.DataFrame(grid, columns=[f"pi_{source}" for source in source_names])
    for index in range(folds):
        candidates[f"fold_{index}_metric"] = fold_metrics[index]
    candidates["mean_metric"] = mean_metric
    candidates["selected"] = np.arange(len(grid)) == best
    candidates["cv_metric"] = ("heldout_nagelkerke_incremental_r2" if family == "binomial"
                               else "heldout_1_minus_full_mse_over_covariate_mse")
    return grid[best].copy(), candidates
