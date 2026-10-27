"""Penalized linear and logistic regression for BLocPRS.

Ridge regularization acts on regional coefficients; the intercept, covariate
effects and genome-wide source coefficients remain unpenalized. Model inputs
are processed in participant blocks to limit memory use.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg.blas import dsyrk
from scipy.optimize import minimize
from scipy.special import expit, logit


# Logistic optimizer iteration limits and stopping tolerances.
BINARY_OPTIONS = {"maxiter": 1000, "maxls": 50, "ftol": 1e-9, "gtol": 1e-5}


@dataclass
class FitResult:
    """Fitted coefficients, objective value, and convergence information."""

    coef: np.ndarray
    objective: float
    converged: bool
    iterations: int
    message: str


@dataclass
class GaussianStatistics:
    """Averaged X.T @ X and X.T @ y used for continuous-model fitting."""

    gram: np.ndarray
    rhs: np.ndarray
    n_samples: int
    design_id: int


# Build the ridge penalty vector for regional coefficients only.
def _penalty(design, lambda_, n_unpenalized):
    p = design.n_features

    if (isinstance(n_unpenalized, (bool, np.bool_))
        or not isinstance(n_unpenalized, (int, np.integer))
        or not 1 <= n_unpenalized <= p):
        raise ValueError("Invalid number of unpenalized coefficients.")

    lambda_ = float(lambda_)

    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("Regional penalty must be finite and nonnegative.")

    # Leave the intercept, covariates, and global source effects unpenalized.
    penalty = np.zeros(p, dtype=np.float64)
    penalty[n_unpenalized:] = lambda_
    return penalty


# Read training blocks and check phenotypes, dimensions, and participant counts.
def _blocks(design):
    n, p = design.n_samples, design.n_features
    seen = 0

    for X, y in design.iter_blocks():
        if y is None:
            raise ValueError("Fitting requires outcomes.")

        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != p or y.shape != (len(X),) or len(X) == 0:
            raise ValueError("Invalid design block dimensions.")

        seen += len(X)
        yield X, y

    if seen != n:
        raise ValueError(f"Design yielded {seen} rows; expected {n}.")


# Accumulate X.T @ X / n and X.T @ y / n without a full n x p design.
def continuous_statistics(design):
    n, p = design.n_samples, design.n_features
    gram = np.zeros((p, p), dtype=np.float64, order="F")
    rhs = np.zeros(p, dtype=np.float64)

    for X, y in _blocks(design):

        # Update one triangle in place to avoid a p x p temporary per block.
        updated = dsyrk(1.0, np.asfortranarray(X.T), beta=1.0, c=gram, trans=0, lower=1, overwrite_c=1)

        if not np.shares_memory(updated, gram):
            raise RuntimeError("BLAS did not update the preallocated Gram matrix in place.")

        rhs += X.T @ y

    # Mirror the accumulated lower triangle before solving the full system.
    for column in range(p):
        gram[column, column + 1:] = gram[column + 1:, column]

    gram /= n
    rhs /= n

    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("Nonfinite Gaussian cross-products.")

    return GaussianStatistics(gram, rhs, n, id(design))


# Compute the full mean-loss objective and gradient, including ridge penalties.
def _objective_gradient(design, coef, penalty, family):
    n = design.n_samples
    loss = 0.0
    gradient = np.zeros_like(coef)

    for X, y in _blocks(design):
        linear_predictor = X @ coef

        if family == "gaussian":
            residual = linear_predictor - y
            loss += 0.5 * float(residual @ residual)
            gradient += X.T @ residual
        else:

            # logaddexp evaluates the logistic loss without exponentiating large logits.
            loss += float(np.sum(np.logaddexp(0.0, linear_predictor) - y * linear_predictor))
            gradient += X.T @ (expit(linear_predictor) - y)

    # Average the data loss; add lambda / 2 * ||eta||^2 only once.
    loss = loss / n + 0.5 * float(np.sum(penalty * coef * coef))
    gradient = gradient / n + penalty * coef

    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite objective or gradient.")

    return float(loss), gradient


# Fit the penalized Gaussian or logistic model, including its intercept.
def fit(design, family, lambda_, n_unpenalized, initial=None, statistics=None):
    if family not in ("gaussian", "binomial"):
        raise ValueError("family must be 'gaussian' or 'binomial'.")

    n, p = design.n_samples, design.n_features
    penalty = _penalty(design, lambda_, n_unpenalized)

    if family == "gaussian":
        if initial is not None:
            raise ValueError("Gaussian fitting does not use initial coefficients.")

        stats = continuous_statistics(design) if statistics is None else statistics

        if (not isinstance(stats, GaussianStatistics) or stats.n_samples != n
            or stats.design_id != id(design) or stats.gram.shape != (p, p)
            or stats.rhs.shape != (p,) or not np.isfinite(stats.gram).all()
            or not np.isfinite(stats.rhs).all()):
            raise ValueError("Gaussian statistics do not match the design.")

        # Add regional ridge penalties to the normal-equation diagonal.
        system = stats.gram.copy(order="F")
        system.flat[::p + 1] += penalty
        message = "numpy.linalg.solve"

        try:
            coef = np.linalg.solve(system, stats.rhs)
        except np.linalg.LinAlgError:

            # Retain the penalty when covariates/source scores are redundant.
            warnings.warn("Singular Gaussian system; solving the penalized normal equations "
                          "by least squares.", RuntimeWarning, stacklevel=2)
            coef = np.linalg.lstsq(system, stats.rhs, rcond=None)[0]
            message = "numpy.linalg.lstsq(penalized normal equations)"

        if not np.isfinite(coef).all():
            raise FloatingPointError("Nonfinite Gaussian coefficients.")

        objective, gradient = _objective_gradient(design, coef, penalty, family)
        tolerance = 1e-7 * max(1.0, float(np.max(np.abs(stats.rhs))))

        if float(np.max(np.abs(gradient))) > tolerance:
            raise RuntimeError("Gaussian solution failed the penalized normal-equation check.")

        return FitResult(coef, objective, True, 1, message)

    if statistics is not None:
        raise ValueError("Gaussian cross-products cannot be used for logistic fitting.")

    # Count cases across all blocks before initializing the logistic intercept.
    cases = 0
    for _, y in _blocks(design):
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("Binomial outcomes must be encoded as zero and one.")
        cases += int(y.sum())
    if not 0 < cases < n:
        raise ValueError("Binomial fitting requires both outcome classes.")

    # Initialize the intercept from prevalence, clipping to keep the logit finite.
    start = np.zeros(p, dtype=np.float64)
    start[0] = float(logit(np.clip(cases / n, 1e-6, 1.0 - 1e-6)))

    if initial is not None:
        initial = np.asarray(initial, dtype=np.float64)

        if initial.shape != (p,) or not np.isfinite(initial).all():
            raise ValueError("Invalid logistic warm-start coefficients.")

        start = initial.copy()

    # Supply the mean logistic loss and its analytic gradient to L-BFGS-B.
    optimum = minimize(lambda coef: _objective_gradient(design, coef, penalty, family), start,
                       jac=True, method="L-BFGS-B", options=dict(BINARY_OPTIONS))

    coef = np.asarray(optimum.x, dtype=np.float64)

    if not optimum.success or not np.isfinite(coef).all():
        raise RuntimeError(f"Logistic fit did not converge: {optimum.message}")

    objective, _ = _objective_gradient(design, coef, penalty, family)
    return FitResult(coef, objective, True, int(optimum.nit), str(optimum.message))
