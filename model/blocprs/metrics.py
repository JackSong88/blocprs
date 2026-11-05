"""Covariate-adjusted evaluation of fixed genetic scores s_i.

Incremental R^2 and covariate-conditioned Nagelkerke R^2 compare regressions
fitted with and without s_i on the same evaluation participants. The intercept,
covariate coefficients and one score coefficient are refitted; the source
and regional weights defining s_i remain fixed.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


class NonEstimableError(ValueError):
    """The evaluation data are valid, but the metric cannot be estimated."""


# Checks and copies evaluation phenotypes and covariates in the supplied row order.
def _inputs(y, covariates):
    y = np.asarray(y, dtype=np.float64)

    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Evaluation requires a finite phenotype vector.")

    if covariates is None:
        covariates = np.empty((len(y), 0), dtype=np.float64)
    else:
        covariates = np.asarray(covariates, dtype=np.float64)

    if covariates.ndim != 2 or covariates.shape[0] != len(y) or not np.isfinite(covariates).all():
        raise ValueError("Covariates must be finite with one row per participant.")

    if len(y) < 2:
        raise NonEstimableError("Evaluation requires at least two participants.")

    return y.copy(), covariates.copy()


# Rescales the combined PRS for numerical stability in evaluation regressions.
def _standardized_score(genetic_score, y):
    score = np.asarray(genetic_score, dtype=np.float64)

    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("Genetic score must be a finite vector matching the participants.")

    # A constant score contributes nothing beyond the intercept.
    if np.all(score == score[0]):
        return np.zeros_like(score)

    # Training standardizes the source PRSs, but their weighted sum s_i can still
    # have a very small SD. Rescaling s_i here improves numerical stability.
    # Uses this evaluation sample's mean and SD; the genetic weights remain fixed.
    # The unpenalized regression's free intercept and score coefficient absorb
    # this rescaling, preserving fitted values and R^2 in exact arithmetic.
    score = score - score.mean()
    score_sd = score.std(ddof=0)

    if not np.isfinite(score_sd) or score_sd <= 0:
        raise NonEstimableError("Genetic score SD is not finite and positive.")

    return score / score_sd


class IncrementalR2:
    """Gain in OLS R^2 from adding fixed s_i to a covariate-only model.

    Null: y ~ intercept + covariates
    Full: y ~ intercept + covariates + frozen PRS

    Both OLS regressions use the same evaluation participants and total phenotype variance.
    The genetic weights defining the PRS remain fixed.

    Incremental R^2 = (RSS_null - RSS_full) / TSS, where TSS = sum_i (y_i - mean(y))^2.
    """

    # Fits the covariate-only OLS model and retains its residual sum of squares.
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

    # Refits OLS with frozen s_i and returns its incremental R^2.
    def score(self, genetic_score):
        score = _standardized_score(genetic_score, self.y)

        # A constant score adds no information beyond the intercept.
        if np.all(score == score[0]):
            return 0.0

        full_design = np.column_stack((self.null_design, score))
        coef = np.linalg.lstsq(full_design, self.y, rcond=None)[0]
        residual = self.y - full_design @ coef

        # Divides the reduction in squared error by total phenotype variation.
        value = float((self.null_ss - residual @ residual) / self.total_ss)

        if not np.isfinite(value):
            raise NonEstimableError("Incremental R^2 is not finite.")

        return value


class NagelkerkeR2:
    """Covariate-conditioned Nagelkerke R^2, following Ge et al. (2019).

    Null: y ~ intercept + covariates
    Full: y ~ intercept + covariates + frozen PRS

    Both regressions are fitted on the same evaluation participants.
    The genetic weights defining the PRS remain fixed.

    ll0 and ll1 are the fitted null and full log-likelihoods:
    R^2 = (1 - exp(-2 * (ll1 - ll0) / n)) / (1 - exp(2 * ll0 / n)).
    """

    # Fits the covariate-only logistic model and computes the normalization factor.
    def __init__(self, y, covariates=None):
        self.y, covariates = _inputs(y, covariates)
        classes = np.unique(self.y)

        if not np.isin(classes, [0.0, 1.0]).all():
            raise ValueError("Binary phenotypes must be coded as 0 and 1.")

        if len(classes) != 2:
            raise NonEstimableError("Nagelkerke R^2 requires both cases and controls.")

        self.covariates = pd.DataFrame(covariates)

        # Removes constant covariates before fitting either logistic model.
        self.covariates = self.covariates.loc[:, self.covariates.var(axis=0) > 0]
        self.null_ll = self._fit_log_likelihood(self.covariates, "covariate-only")

        self.denominator = 1.0 - np.exp(2.0 * self.null_ll / len(self.y))

        if not np.isfinite(self.denominator) or self.denominator <= 0:
            raise NonEstimableError("Nagelkerke normalization is not estimable.")

    # Returns the logistic log-likelihood only when the fit converges and is finite.
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

    # Refits logistic regression with frozen s_i and returns Nagelkerke R^2.
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


# Evaluates fixed s_i using OLS regressions fitted on these participants.
def incremental_r2(y, genetic_score, covariates=None):
    return IncrementalR2(y, covariates).score(genetic_score)


# Evaluates fixed s_i using logistic regressions fitted on these participants.
def nagelkerke_r2(y, genetic_score, covariates=None):
    return NagelkerkeR2(y, covariates).score(genetic_score)
