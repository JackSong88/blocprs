"""Covariate-adjusted evaluation of fixed polygenic scores.

Incremental R^2 and covariate-conditioned Nagelkerke R^2 compare regressions
fitted with and without the PRS on the same evaluation participants.
Source and regional coefficients remain fixed during evaluation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


INCREMENTAL_R2_VERSION = "incremental_r2_ols_v1"
INCREMENTAL_R2_DEFINITION = "R2_OLS(y~1+C+genetic_score)-R2_OLS(y~1+C);validation_refit"
NAGELKERKE_R2_VERSION = "nagelkerke_r2_logit_v1"
NAGELKERKE_R2_DEFINITION = ("(1-exp(-2*(ll_Logit(y~1+C+genetic_score)-ll_Logit(y~1+C))/n))"
                          "/(1-exp(2*ll_Logit(y~1+C)/n));validation_refit")


class NonEstimableError(ValueError):
    """The evaluation data are valid, but the metric cannot be estimated."""


# Check and copy evaluation phenotypes and covariates in the supplied row order.
def _inputs(y, covariates):
    y = np.asarray(y, dtype=np.float64)

    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Evaluation requires a finite phenotype vector.")

    covariates = (np.empty((len(y), 0), dtype=np.float64) if covariates is None
                  else np.asarray(covariates, dtype=np.float64))

    if covariates.ndim != 2 or covariates.shape[0] != len(y) or not np.isfinite(covariates).all():
        raise ValueError("Covariates must be finite with one row per participant.")

    if len(y) < 2:
        raise NonEstimableError("Evaluation requires at least two participants.")

    return y.copy(), covariates.copy()


# Standardize the frozen score for the unpenalized evaluation regression.
def _standardized_score(genetic_score, y):
    score = np.asarray(genetic_score, dtype=np.float64)

    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("Genetic score must be a finite vector matching the participants.")

    # A constant score contributes nothing beyond the intercept.
    if np.all(score == score[0]):
        return np.zeros_like(score)

    # Rescale only this auxiliary regression's input for numerical stability.
    # With a free intercept and score coefficient, its fitted values are unchanged in exact arithmetic.
    score = score - score.mean()
    score_sd = score.std(ddof=0)

    if not np.isfinite(score_sd) or score_sd <= 0:
        raise NonEstimableError("Genetic score SD is not finite and positive.")

    return score / score_sd


class IncrementalR2:
    """Gain in OLS R^2 from adding the fixed PRS to a covariate-only model.

    Both models use the same participants and total phenotype variance.
    """

    # Fit the covariate-only OLS model and retain its residual sum of squares.
    def __init__(self, y, covariates=None):
        self.y, covariates = _inputs(y, covariates)
        centered = self.y - self.y.mean()
        self.total_ss = float(centered @ centered)

        if not np.isfinite(self.total_ss) or self.total_ss <= 0:
            raise NonEstimableError("Incremental R^2 requires positive phenotype variance.")

        self.null_design = np.column_stack((np.ones(len(self.y)), covariates))
        coef = np.linalg.lstsq(self.null_design, self.y, rcond=None)[0]
        residual = self.y - self.null_design @ coef
        self.null_ss = float(residual @ residual)

    # Refit OLS with the frozen score and return its incremental R^2.
    def score(self, genetic_score):
        score = _standardized_score(genetic_score, self.y)

        # A constant score adds no information beyond the intercept.
        if np.all(score == score[0]):
            return 0.0

        full_design = np.column_stack((self.null_design, score))
        coef = np.linalg.lstsq(full_design, self.y, rcond=None)[0]
        residual = self.y - full_design @ coef

        # Divide the reduction in squared error by total phenotype variation.
        value = float((self.null_ss - residual @ residual) / self.total_ss)

        if not np.isfinite(value):
            raise NonEstimableError("Incremental R^2 is not finite.")

        return value


class NagelkerkeR2:
    """Covariate-conditioned Nagelkerke R^2 for a fixed polygenic score.

    Both models use the same participants; normalization uses the covariate-only fit.
    """

    # Fit the covariate-only logistic model and compute the normalization factor.
    def __init__(self, y, covariates=None):
        self.y, covariates = _inputs(y, covariates)
        classes = np.unique(self.y)

        if not np.isin(classes, [0.0, 1.0]).all():
            raise ValueError("Binary phenotypes must be coded as 0 and 1.")

        if len(classes) != 2:
            raise NonEstimableError("Nagelkerke R^2 requires both cases and controls.")

        self.covariates = pd.DataFrame(covariates)

        # Remove constant covariates before fitting either logistic model.
        self.covariates = self.covariates.loc[:, self.covariates.var(axis=0) > 0]
        self.null_ll = self._fit_log_likelihood(self.covariates, "covariate-only")

        self.denominator = 1.0 - np.exp(2.0 * self.null_ll / len(self.y))

        if not np.isfinite(self.denominator) or self.denominator <= 0:
            raise NonEstimableError("Nagelkerke normalization is not estimable.")

    # Return the logistic log-likelihood only when the fit converges and is finite.
    def _fit_log_likelihood(self, covariates, label):
        try:

            # This unpenalized fit estimates evaluation calibration, not new PRS combination weights.
            result = sm.Logit(self.y, sm.add_constant(covariates)).fit(disp=0)
        except (np.linalg.LinAlgError, PerfectSeparationError, FloatingPointError) as exc:
            raise NonEstimableError(f"Nagelkerke {label} fit is not estimable: {exc}") from exc

        finite = np.isfinite(result.params).all() and np.isfinite(result.llf)

        if not result.mle_retvals.get("converged", False) or not finite:
            raise NonEstimableError(f"Nagelkerke {label} fit did not converge to a finite fit.")

        return float(result.llf)

    # Refit logistic regression with the frozen score and return Nagelkerke R^2.
    def score(self, genetic_score):
        score = _standardized_score(genetic_score, self.y)

        # A constant score adds no information beyond the intercept.
        if np.all(score == score[0]):
            return 0.0

        full_ll = self._fit_log_likelihood(self.covariates.assign(pred_val=score), "covariate-plus-PRS")

        value = (1.0 - np.exp(-2.0 * (full_ll - self.null_ll) / len(self.y))) / self.denominator

        if not np.isfinite(value):
            raise NonEstimableError("Nagelkerke R^2 is not finite.")

        return float(value)


# Evaluate the fixed PRS using OLS regressions fitted on these participants.
def incremental_r2(y, genetic_score, covariates=None):
    return IncrementalR2(y, covariates).score(genetic_score)


# Evaluate the fixed PRS using logistic regressions fitted on these participants.
def nagelkerke_r2(y, genetic_score, covariates=None):
    return NagelkerkeR2(y, covariates).score(genetic_score)
