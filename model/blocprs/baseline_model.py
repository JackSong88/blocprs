"""Single-source PRS and genome-wide MultiPRS models.

MultiPRS combines source scores using unregularized linear or logistic
regression, with optional regularization or nonnegative source coefficients.
The intercept and covariate effects remain unpenalized. Regularization
parameters are selected by cross-validation using the same covariate-adjusted
metrics as evaluation.
"""

import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, StratifiedKFold

from .cv import _positive_integer, _subsample_indices
from .data import PRSData
from .metrics import IncrementalR2, NagelkerkeR2


# Select one named source without choosing it using evaluation outcomes.
def select_source(data, source):
    index = data.source_names.index(source)
    return PRSData(data.window_scores[:, :, index:index + 1], [source], data.window_names,
                   y=data.y, covariates=data.covariates, covariate_names=data.covariate_names,
                   sample_ids=data.sample_ids)


# Explicitly sum raw window scores when a genome-wide dataset is unavailable.
def genomewide_data(data, block_size=8192):
    block_size = _positive_integer(block_size, "block_size")
    scores = np.empty((data.n_samples, 1, data.n_sources), dtype=np.float64)

    for start in range(0, data.n_samples, block_size):
        rows = slice(start, min(start + block_size, data.n_samples))
        scores[rows, 0] = np.sum(data.window_scores[rows], axis=1, dtype=np.float64)

    return PRSData(scores, data.source_names, ["genomewide"], y=data.y,
                   covariates=data.covariates, covariate_names=data.covariate_names,
                   sample_ids=data.sample_ids)


@dataclass
class _SourceScaler:
    """Training means and population SDs of source scores; covariates and y remain unscaled."""

    source_mean: np.ndarray
    source_sd: np.ndarray

    # Compute source scaling only on the supplied fitting participants.
    @classmethod
    def fit(cls, sources):
        sd = np.std(sources, axis=0, ddof=0)
        sd[sd == 0] = 1
        return cls(np.mean(sources, axis=0), sd)

    # Apply the saved training means and SDs to new participants.
    def transform(self, sources):
        return (sources - self.source_mean) / self.source_sd


# Remove the intercept/covariate component before continuous source fitting.
def _residuals(nuisance, sources, y):
    values = np.column_stack((y, sources))
    residual = values - nuisance @ np.linalg.lstsq(nuisance, values, rcond=None)[0]
    return residual[:, 1:], residual[:, 0]


# Fit the specified objective, leaving intercept and covariates unpenalized.
# Data loss: 0.5 * mean squared error for continuous traits; mean logistic loss for binary traits.
def _fit_once(nuisance, sources, y, family, variant, lambda_, ratio=0.0,
              initial=None, residuals=None, null_fit=None):
    n, k = sources.shape
    p = nuisance.shape[1]
    l1 = lambda_ * ratio
    l2 = lambda_ * (1 - ratio)

    if family == "gaussian":
        X_res, y_res = _residuals(nuisance, sources, y) if residuals is None else residuals
        iterations = 0

        if variant == "nonnegative":
            gamma = nnls(X_res, y_res)[0]
        elif lambda_ == 0:
            gamma = np.linalg.lstsq(X_res, y_res, rcond=None)[0]
        elif l1 == 0:
            gram = X_res.T @ X_res / n
            gamma = np.linalg.solve(gram + l2 * np.eye(k), X_res.T @ y_res / n)
        else:
            fitted = ElasticNet(alpha=lambda_, l1_ratio=ratio, fit_intercept=False,
                                max_iter=20000, tol=1e-8)

            # A failed optimizer is not accepted as a successful comparator fit.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fitted.fit(X_res, y_res)

            gamma = fitted.coef_
            iterations = int(fitted.n_iter_)

        alpha = np.linalg.lstsq(nuisance, y - sources @ gamma, rcond=None)[0]
        coef = np.concatenate((alpha, gamma))
        residual = y - nuisance @ alpha - sources @ gamma
        objective = 0.5 * float(residual @ residual) / n
        objective += l1 * float(np.abs(gamma).sum()) + 0.5 * l2 * float(gamma @ gamma)
        return coef, {"objective": objective, "iterations": iterations, "converged": True}

    # At the L1 zero-solution bound, return the fitted null model exactly.
    if l1 > 0:
        null_coef, null_info, max_gradient = (_binomial_null(nuisance, sources, y)
                                             if null_fit is None else null_fit)

        if lambda_ >= max_gradient / ratio:
            return np.concatenate((null_coef, np.zeros(k))), dict(null_info)

    design = np.column_stack((nuisance, sources))
    initial = np.zeros(p + k) if initial is None else np.asarray(initial, dtype=float).copy()

    if not np.any(initial):
        initial[0] = logit(np.clip(y.mean(), 1e-6, 1 - 1e-6))

    # Positive/negative parts give an exact L1 penalty with nonnegative bounds.
    if l1 > 0:
        start = np.concatenate((initial[:p], np.maximum(initial[p:], 0), np.maximum(-initial[p:], 0)))

        def objective(params):
            positive, negative = params[p:p + k], params[p + k:]
            gamma = positive - negative
            linear = nuisance @ params[:p] + sources @ gamma
            residual = expit(linear) - y
            value = np.mean(np.logaddexp(0, linear) - y * linear)
            value += l1 * (positive.sum() + negative.sum()) + 0.5 * l2 * (gamma @ gamma)
            gradient = sources.T @ residual / n + l2 * gamma
            return float(value), np.concatenate((nuisance.T @ residual / n, gradient + l1, -gradient + l1))

        bounds = [(None, None)] * p + [(0, None)] * (2 * k)
    else:
        start = initial

        def objective(params):
            linear = design @ params
            gamma = params[p:]
            value = np.mean(np.logaddexp(0, linear) - y * linear) + 0.5 * l2 * (gamma @ gamma)
            gradient = design.T @ (expit(linear) - y) / n
            gradient[p:] += l2 * gamma
            return float(value), gradient

        bounds = [(None, None)] * p + [(0, None)] * k if variant == "nonnegative" else None

    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1000 if l1 > 0 else 2000, "maxls": 50,
                               "ftol": 1e-10, "gtol": 1e-6})

    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"MultiPRS logistic fit did not converge: {result.message}")

    coef = (np.concatenate((result.x[:p], result.x[p:p + k] - result.x[p + k:]))
            if l1 > 0 else result.x)
    return coef, {"objective": float(result.fun), "iterations": int(result.nit), "converged": True}


# Fit the binary covariate-only optimum and its maximum source gradient.
def _binomial_null(nuisance, sources, y):
    empty = np.empty((len(y), 0))
    coef, info = _fit_once(nuisance, empty, y, "binomial", "unregularized", 0.0)
    gradient = sources.T @ (expit(nuisance @ coef) - y) / len(y)
    return coef, info, float(np.max(np.abs(gradient)))


# Find the penalty at which all source coefficients can be zero.
def _lambda_max(nuisance, sources, y, family, ratio, residuals=None, null_fit=None):
    if family == "gaussian":
        X_res, y_res = _residuals(nuisance, sources, y) if residuals is None else residuals
        max_gradient = float(np.max(np.abs(X_res.T @ y_res / len(y))))
    else:
        _, _, max_gradient = _binomial_null(nuisance, sources, y) if null_fit is None else null_fit

    value = max_gradient / ratio
    return value if value > 0 else 1.0


class MultiPRS:
    """Genome-wide PRS combination by linear or logistic regression.

    Input PRSData must contain one genome-wide column per source (W=1).
    Use genomewide_data explicitly to construct sums from window inputs.
    Covariates and phenotypes remain in their supplied units. Source scores
    use training-fold means and SDs during CV and full-training values for refitting.

    lambda_ fixes an absolute penalty and skips CV. Otherwise ridge searches
    25 absolute values from 1e-6 to 1e4; lasso/elasticnet search 25 fractions
    from 1 to 1e-4 of each fold's lambda_max. lambda_grid overrides those values.
    Elastic-net CV uses L1 ratios 0.1, 0.5 and 0.9; fixed-lambda fits default to
    ratio 0.5. Unregularized and nonnegative fits have no penalty search.
    """

    # Set the family, comparator variant and CV options.
    def __init__(self, family="gaussian", *, variant="unregularized", lambda_=None,
                 lambda_grid=None, l1_ratios=None, cv_folds=5, cv_max_samples=50000,
                 random_state=42, block_size=8192):
        self.family, self.variant = family, variant
        self.lambda_, self.lambda_grid, self.l1_ratios = lambda_, lambda_grid, l1_ratios
        self.cv_folds, self.cv_max_samples = cv_folds, cv_max_samples
        self.random_state, self.block_size = random_state, block_size

    # Check the score layout and finite inputs without guessing window aggregation.
    def _arrays(self, data, training=False):
        if not isinstance(data, PRSData) or data.n_windows != 1:
            raise ValueError("MultiPRS requires PRSData with one genome-wide score per source.")

        sources = np.asarray(data.window_scores[:, 0], dtype=np.float64)
        covariates = np.asarray(data.covariates, dtype=np.float64)
        y = None if data.y is None else np.asarray(data.y, dtype=np.float64)

        if not np.isfinite(sources).all() or not np.isfinite(covariates).all():
            raise ValueError("MultiPRS scores and covariates must be finite.")

        if training:
            if y is None or not np.isfinite(y).all():
                raise ValueError("Training requires finite phenotypes.")

            if self.family == "binomial" and not np.array_equal(np.unique(y), [0, 1]):
                raise ValueError("Binary training requires both classes, coded as 0 and 1.")

        return sources, covariates, y

    # Resolve explicit penalty choices without applying BLocPRS's regional grid.
    def _choices(self):
        if self.family not in ("gaussian", "binomial"):
            raise ValueError("family must be 'gaussian' or 'binomial'.")

        if self.variant not in ("unregularized", "ridge", "lasso", "elasticnet", "nonnegative"):
            raise ValueError("Unknown MultiPRS variant.")

        self.block_size = _positive_integer(self.block_size, "block_size")
        if self.variant in ("unregularized", "nonnegative"):
            if self.lambda_ not in (None, 0) or self.lambda_grid is not None or self.l1_ratios is not None:
                raise ValueError("Unregularized/nonnegative models have no penalty grid or L1 ratio.")

            return np.array([0.0]), [0.0], False

        if self.lambda_ is not None and self.lambda_grid is not None:
            raise ValueError("Choose a fixed lambda_ or lambda_grid, not both.")

        search = self.lambda_ is None
        if not search:
            candidates = np.array([self.lambda_], dtype=float)
        elif self.lambda_grid is not None:
            candidates = np.asarray(self.lambda_grid, dtype=float)
        else:
            candidates = np.logspace(-6, 4, 25) if self.variant == "ridge" else np.logspace(0, -4, 25)

        if (candidates.ndim != 1 or len(candidates) == 0 or not np.isfinite(candidates).all()
            or np.any(candidates < 0) or len(np.unique(candidates)) != len(candidates)):
            raise ValueError("Penalties must be distinct, finite, nonnegative values.")

        if self.variant == "elasticnet":
            ratios = ([0.1, 0.5, 0.9] if search else [0.5]) if self.l1_ratios is None else list(self.l1_ratios)
        else:
            if self.l1_ratios is not None:
                raise ValueError("l1_ratios is only used by elasticnet.")

            ratios = [1.0 if self.variant == "lasso" else 0.0]

        if not ratios or any(not np.isfinite(r) or r <= 0 or r > 1 for r in ratios if self.variant == "elasticnet"):
            raise ValueError("Elastic-net L1 ratios must lie in (0, 1].")

        if not search and len(ratios) != 1:
            raise ValueError("A fixed lambda_ requires exactly one elastic-net L1 ratio.")

        return candidates, ratios, search

    # Tune on training folds, then fit every coefficient on all training rows.
    def fit(self, data):
        candidates, ratios, search = self._choices()
        sources, covariates, y = self._arrays(data, training=True)
        nuisance = np.column_stack((np.ones(len(y)), covariates))
        selected, ratio = float(candidates[0]), float(ratios[0])
        cv_results = []

        if search:
            selected, ratio, cv_results = self._cross_validate(sources, covariates, y, candidates, ratios)

        scaler = _SourceScaler.fit(sources)
        standardized = scaler.transform(sources)
        residuals = _residuals(nuisance, standardized, y) if self.family == "gaussian" else None
        null_fit = (_binomial_null(nuisance, standardized, y)
                    if self.family == "binomial" and self.variant in ("lasso", "elasticnet") else None)
        lambda_ = selected

        if search and self.variant in ("lasso", "elasticnet"):
            lambda_ *= _lambda_max(nuisance, standardized, y, self.family, ratio, residuals, null_fit)

        coef, fit_info = _fit_once(nuisance, standardized, y, self.family, self.variant,
                                  lambda_, ratio, residuals=residuals, null_fit=null_fit)

        # Replace a previous fitted state only after the new fit succeeds.
        self.scaler_, self.coef_, self.fit_info_ = scaler, coef, fit_info
        self.cv_results_ = cv_results
        self.source_names_, self.window_names_ = data.source_names, data.window_names
        self.covariate_names_ = data.covariate_names
        self.selected_candidate_, self.selected_lambda_ = selected, float(lambda_)
        self.selected_l1_ratio_ = ratio
        self.n_training_samples_ = data.n_samples
        return self

    # Refit source scaling and the comparator independently within every CV fold.
    def _cross_validate(self, sources, covariates, y, candidates, ratios):
        folds = _positive_integer(self.cv_folds, "cv_folds", minimum=2)
        cap = len(y) if self.cv_max_samples is None else _positive_integer(self.cv_max_samples, "cv_max_samples", folds)
        total = min(len(y), cap) // folds * folds

        if total < folds:
            raise ValueError("Too few participants for CV.")

        sampled = _subsample_indices(y, total, np.random.RandomState(self.random_state),
                                    stratify=self.family == "binomial", min_per_class=folds)
        if self.family == "binomial" and np.min(np.unique(y[sampled], return_counts=True)[1]) < folds:
            raise ValueError("Binary CV needs at least cv_folds participants in each class.")

        splitter_type = StratifiedKFold if self.family == "binomial" else KFold
        splitter = splitter_type(n_splits=folds, shuffle=True, random_state=self.random_state)
        results = [{"candidate": float(value), "l1_ratio": float(ratio), "fold_scores": []}
                   for ratio in ratios for value in candidates]
        metric_type = NagelkerkeR2 if self.family == "binomial" else IncrementalR2

        for train, validation in splitter.split(sampled, y[sampled]):
            train, validation = sampled[train], sampled[validation]
            scaler = _SourceScaler.fit(sources[train])
            S_train, S_val = scaler.transform(sources[train]), scaler.transform(sources[validation])
            nuisance = np.column_stack((np.ones(len(train)), covariates[train]))
            residuals = _residuals(nuisance, S_train, y[train]) if self.family == "gaussian" else None
            null_fit = (_binomial_null(nuisance, S_train, y[train])
                        if self.family == "binomial" and self.variant in ("lasso", "elasticnet") else None)
            metric = metric_type(y[validation], covariates[validation])
            initial, previous_ratio, lambda_max = None, None, 1.0

            for entry in results:
                ratio = entry["l1_ratio"]
                if ratio != previous_ratio:
                    initial = None
                    if self.variant in ("lasso", "elasticnet"):
                        lambda_max = _lambda_max(nuisance, S_train, y[train], self.family, ratio, residuals, null_fit)
                    previous_ratio = ratio

                lambda_ = entry["candidate"] * lambda_max
                coef, _ = _fit_once(nuisance, S_train, y[train], self.family, self.variant,
                                    lambda_, ratio, initial=initial, residuals=residuals, null_fit=null_fit)
                entry["fold_scores"].append(float(metric.score(S_val @ coef[nuisance.shape[1]:])))
                initial = coef

        for entry in results:
            entry["mean_score"] = float(np.mean(entry["fold_scores"]))

        selected = max(results, key=lambda entry: entry["mean_score"])
        return selected["candidate"], selected["l1_ratio"], results

    # Return the fitted intercept.
    @property
    def alpha_0_(self):
        return float(self.coef_[0])

    # Return the coefficients of the raw covariates.
    @property
    def alpha_(self):
        return self.coef_[1:1 + len(self.covariate_names_)].copy()

    # Return one coefficient per standardized genome-wide source.
    @property
    def gamma_(self):
        return self.coef_[1 + len(self.covariate_names_):].copy()

    # Name the comparator without using its evaluation performance.
    @property
    def method_name_(self):
        return "MultiPRS" if self.variant == "unregularized" else f"MultiPRS-{self.variant}"

    # Compute genetic-only scores using saved source scaling and coefficients.
    def predict_prs(self, data):
        sources, _, _ = self._arrays(data)

        if data.source_names != self.source_names_ or data.window_names != self.window_names_:
            raise ValueError("Source or genome-wide column names/order differ from training.")

        return self.scaler_.transform(sources) @ self.gamma_

    # Add the fitted raw-covariate contribution and intercept to the genetic score.
    def predict_linear(self, data):
        if data.covariate_names != self.covariate_names_:
            raise ValueError("Covariate names/order differ from training.")

        return self.alpha_0_ + data.covariates @ self.alpha_ + self.predict_prs(data)

    # Convert the fitted full logistic predictor to event probabilities.
    def predict_proba(self, data):
        if self.family != "binomial":
            raise ValueError("predict_proba requires a binomial model.")

        return expit(self.predict_linear(data))

    # Evaluate the frozen genetic score with the same metric used by BLocPRS.
    def score(self, data):
        if data.covariate_names != self.covariate_names_:
            raise ValueError("Evaluation covariate names/order differ from training.")

        metric = NagelkerkeR2 if self.family == "binomial" else IncrementalR2
        return metric(data.y, data.covariates).score(self.predict_prs(data))

    # Save fitted parameters and CV results in one archive, without training data.
    def save(self, path):
        path = Path(path)
        metadata = {"format": "blocprs-baseline-v1", "family": self.family, "variant": self.variant,
                    "source": getattr(self, "source", None), "source_names": self.source_names_,
                    "window_names": self.window_names_, "covariate_names": self.covariate_names_,
                    "selected_candidate": self.selected_candidate_, "selected_lambda": self.selected_lambda_,
                    "selected_l1_ratio": self.selected_l1_ratio_, "cv_results": self.cv_results_,
                    "cv_metric": "nagelkerke_r2_logit_v1" if self.family == "binomial" else "incremental_r2_ols_v1",
                    "scaling_scope": "fold_training_sources_only", "loss_reduction": "mean",
                    "fit_info": self.fit_info_, "n_training_samples": self.n_training_samples_,
                    "parameters": {"lambda_": self.lambda_,
                                   "lambda_grid": None if self.lambda_grid is None else list(map(float, self.lambda_grid)),
                                   "l1_ratios": None if self.l1_ratios is None else list(map(float, self.l1_ratios)),
                                   "cv_folds": self.cv_folds, "cv_max_samples": self.cv_max_samples,
                                   "random_state": self.random_state, "block_size": self.block_size}}
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)

        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
                                    coef=self.coef_, source_mean=self.scaler_.source_mean,
                                    source_sd=self.scaler_.source_sd)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    # Restore a MultiPRS or single-source fit from a filename or binary stream.
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))

            if metadata.get("format") != "blocprs-baseline-v1":
                raise ValueError("Unsupported baseline model archive.")

            if metadata["source"] is None:
                model = cls(metadata["family"], variant=metadata["variant"], **metadata["parameters"])
            else:
                model = SingleSourcePRS(metadata["source"], metadata["family"],
                                         block_size=metadata["parameters"]["block_size"])

            model.source_names_ = tuple(metadata["source_names"])
            model.window_names_ = tuple(metadata["window_names"])
            model.covariate_names_ = tuple(metadata["covariate_names"])
            model.coef_ = saved["coef"].astype(float, copy=True)
            mean, sd = saved["source_mean"].copy(), saved["source_sd"].copy()
            k = len(model.source_names_)

            if (model.coef_.shape != (1 + len(model.covariate_names_) + k,)
                or mean.shape != (k,) or sd.shape != (k,) or np.any(sd <= 0)
                or not all(np.isfinite(value).all() for value in (model.coef_, mean, sd))):
                raise ValueError("Invalid saved baseline coefficients or source scaling.")

            model.scaler_ = _SourceScaler(mean, sd)
            model.selected_candidate_ = metadata["selected_candidate"]
            model.selected_lambda_ = metadata["selected_lambda"]
            model.selected_l1_ratio_ = metadata["selected_l1_ratio"]
            model.cv_results_, model.fit_info_ = metadata["cv_results"], metadata["fit_info"]
            model.n_training_samples_ = metadata["n_training_samples"]
        return model


class SingleSourcePRS(MultiPRS):
    """Unregularized regression for one prespecified genome-wide PRS."""

    # Record the requested source and use unregularized OLS/logistic fitting.
    def __init__(self, source, family="gaussian", *, block_size=8192):
        super().__init__(family, block_size=block_size)
        self.source = source

    # Fit only the named source using the shared baseline implementation.
    def fit(self, data):
        return super().fit(select_source(data, self.source))

    # Extract the same source from a complete source panel during prediction.
    def predict_prs(self, data):
        return super().predict_prs(select_source(data, self.source))

    # Identify the supplied source without implying it was selected on test data.
    @property
    def method_name_(self):
        return f"Single-source:{self.source}"
