"""Evaluate how much a fixed genetic score adds beyond covariates.

Regressions are fitted on the evaluation participants, with and without the
genetic score. The source and window coefficients that produced it stay fixed.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


INCREMENTAL_R2_VERSION = "incremental_r2_ols_v1"
INCREMENTAL_R2_DEFINITION = "R2_OLS(y~1+C+genetic_score)-R2_OLS(y~1+C);validation_refit"
NAGELKERKE_R2_VERSION = "nagelkerke_r2_logit_v1"
NAGELKERKE_R2_DEFINITION = ("(1-exp(-2*(ll_Logit(y~1+C+genetic_score)-ll_Logit(y~1+C))/n))"
                          "/(1-exp(2*ll_Logit(y~1+C)/n));validation_refit")


# Check and copy evaluation phenotypes and covariates in the supplied row order.
def _inputs(y, covariates):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Evaluation requires at least two finite outcomes.")

    covariates = (np.empty((len(y), 0), dtype=np.float64) if covariates is None
                  else np.asarray(covariates, dtype=np.float64))
    if covariates.ndim != 2 or covariates.shape[0] != len(y) or not np.isfinite(covariates).all():
        raise ValueError("Covariates must be finite with one row per outcome.")

    return y.copy(), covariates.copy()


# Require one finite genetic score per evaluation participant.
def _score_vector(genetic_score, y):
    score = np.asarray(genetic_score, dtype=np.float64)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("Genetic score must be a finite vector matching the outcomes.")
    return score


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
            raise ValueError("Incremental R^2 requires positive outcome variance.")

        self.null_design = np.column_stack((np.ones(len(self.y)), covariates))
        coef = np.linalg.lstsq(self.null_design, self.y, rcond=None)[0]
        residual = self.y - self.null_design @ coef
        self.null_ss = float(residual @ residual)

    # Refit OLS with the frozen score and return its incremental R^2.
    def score(self, genetic_score):
        score = _score_vector(genetic_score, self.y)
        full_design = np.column_stack((self.null_design, score))
        coef = np.linalg.lstsq(full_design, self.y, rcond=None)[0]
        residual = self.y - full_design @ coef

        # Divide the reduction in squared error by total phenotype variation.
        value = float((self.null_ss - residual @ residual) / self.total_ss)
        if not np.isfinite(value):
            raise ValueError("Incremental R^2 is not finite.")
        return value  # Unclipped incremental R^2.


class NagelkerkeR2:
    """Nagelkerke R^2 for adding the fixed PRS to a covariate-only logistic model.

    Both models use the same participants; normalization uses the covariate-only fit.
    """

    # Fit the covariate-only logistic model and compute the normalization factor.
    def __init__(self, y, covariates=None):
        self.y, covariates = _inputs(y, covariates)
        if not np.array_equal(np.unique(self.y), [0.0, 1.0]):
            raise ValueError("Nagelkerke R^2 requires binary outcomes with both classes.")

        self.covariates = pd.DataFrame(covariates)

        # Remove constant covariates before fitting either logistic model.
        self.covariates = self.covariates.loc[:, self.covariates.var(axis=0) > 0]
        self.null_ll = self._fit_log_likelihood(self.covariates, "covariate-only")

        self.denominator = 1.0 - np.exp(2.0 * self.null_ll / len(self.y))
        if not np.isfinite(self.denominator) or self.denominator <= 0:
            raise ValueError("Nagelkerke normalization is not estimable.")

    # Return the logistic log-likelihood only when the fit converges and is finite.
    def _fit_log_likelihood(self, covariates, label):
        try:
            result = sm.Logit(self.y, sm.add_constant(covariates)).fit(disp=0)
        except Exception as exc:
            raise ValueError(f"Nagelkerke {label} fit is not estimable: {exc}") from exc

        if (not result.mle_retvals.get("converged", False)
            or not np.isfinite(result.params).all()
            or not np.isfinite(result.llf)):
            raise ValueError(f"Nagelkerke {label} fit did not converge to a finite fit.")

        return float(result.llf)

    # Refit logistic regression with the frozen score and return Nagelkerke R^2.
    def score(self, genetic_score):
        score = _score_vector(genetic_score, self.y)
        full_ll = self._fit_log_likelihood(self.covariates.assign(pred_val=score), "covariate-plus-PRS")

        value = (1.0 - np.exp(-2.0 * (full_ll - self.null_ll) / len(self.y))) / self.denominator
        if not np.isfinite(value):
            raise ValueError("Nagelkerke R^2 is not finite.")
        return float(value)


# Evaluate the fixed PRS using OLS regressions fitted on these participants.
def incremental_r2(y, genetic_score, covariates=None):
    return IncrementalR2(y, covariates).score(genetic_score)


# Evaluate the fixed PRS using logistic regressions fitted on these participants.
def nagelkerke_r2(y, genetic_score, covariates=None):
    return NagelkerkeR2(y, covariates).score(genetic_score)
