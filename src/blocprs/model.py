"""Fit BLocPRS models, generate predictions and save fitted parameters."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit

from .cv import select_lambda
from .data import Design, PRSData, Scaler
from .metrics import IncrementalR2, NagelkerkeR2
from .solvers import fit as solve


class BLocPRS:
    """Fit global and regional PRS coefficients, penalizing only regional effects.

    ``family`` is ``"gaussian"`` or ``"binomial"``. A fixed ``lambda_`` skips
    CV; otherwise select from ``lambda_grid`` (25 log ladder values from 1e-3 to 1e3).
    CV evaluates fixed genetic scores using incremental R^2 for continuous
    phenotypes or Nagelkerke R^2 for binary phenotypes, not training loss.
    Means and SDs are computed once on the whole supplied training set and
    reused across CV folds and for prediction.

    ``block_size`` limits the number of participants processed at a time.
    Continuous fitting still stores a feature-by-feature matrix (X.T @ X).
    Fitted models keep coefficients, training means and SDs, not training data.
    """

    # Set the phenotype family, penalty choices and fitting options.
    def __init__(self, family="gaussian", *, lambda_=None, lambda_grid=None,
                 cv_folds=5, cv_max_samples=50000, random_state=42, block_size=8192):
        self.family = family
        self.lambda_ = lambda_
        self.lambda_grid = lambda_grid
        self.cv_folds = cv_folds
        self.cv_max_samples = cv_max_samples
        self.random_state = random_state
        self.block_size = block_size

    # Select the penalty and fit using training participants only.
    def fit(self, data):
        if not isinstance(data, PRSData) or data.y is None:
            raise ValueError("fit requires PRSData with an outcome vector.")

        if self.lambda_ is not None:
            if not np.isfinite(self.lambda_) or self.lambda_ < 0:
                raise ValueError("lambda_ must be finite and nonnegative.")
            if self.lambda_grid is not None:
                raise ValueError("Choose a fixed lambda_ or a lambda_grid, not both.")

        # Compute training means and SDs once; CV and prediction reuse them.
        scaler = Scaler.fit(data, self.family, self.block_size)

        grid = (np.logspace(-3, 3, 25) if self.lambda_grid is None
                else np.asarray(self.lambda_grid, dtype=float))

        if self.lambda_ is None:
            selected_lambda, cv_results = select_lambda(data, scaler, self.family, grid,
                                                        cv_folds=self.cv_folds,
                                                        cv_max_samples=self.cv_max_samples,
                                                        random_state=self.random_state,
                                                        block_size=self.block_size)
        else:
            selected_lambda, cv_results = float(self.lambda_), []

        # Refit on all training rows, without reusing fold coefficients.
        design = Design(data, scaler, block_size=self.block_size)
        n_unpenalized = 1 + design.n_covariates + len(data.source_names)
        result = solve(design, self.family, selected_lambda, n_unpenalized)

        # Retain fitted parameters and summaries, not the training arrays.
        self.scaler_ = scaler
        self.source_names_ = tuple(data.source_names)
        self.window_names_ = tuple(data.window_names)
        self.covariate_names_ = tuple(data.covariate_names)
        self.coef_ = result.coef.copy()
        self.selected_lambda_ = float(selected_lambda)
        self.cv_results_ = cv_results
        self.n_training_samples_ = int(design.n_samples)
        self.fit_info_ = {
            "objective": float(result.objective),
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "message": str(result.message),
        }

        return self

    # Stop if coefficients have not been fitted or loaded.
    def _require_fitted(self):
        if not hasattr(self, "coef_"):
            raise ValueError("Fit or load a BLocPRS model before using it.")

    # Return alpha_0, the fitted intercept.
    @property
    def alpha_0_(self):
        self._require_fitted()
        return float(self.coef_[0])

    # Return alpha, the coefficients of the transformed covariates.
    @property
    def alpha_(self):
        self._require_fitted()
        return self.coef_[1 : 1 + len(self.covariate_names_)].copy()

    # Return gamma, one global coefficient per source.
    @property
    def gamma_(self):
        self._require_fitted()
        start = 1 + len(self.covariate_names_)
        return self.coef_[start : start + len(self.source_names_)].copy()

    # Return eta, the regional coefficients in W x K order.
    @property
    def eta_(self):
        self._require_fitted()
        start = 1 + len(self.covariate_names_) + len(self.source_names_)
        shape = (len(self.window_names_), len(self.source_names_))
        return self.coef_[start:].reshape(shape).copy()

    # Center eta by removing each source's mean regional coefficient.
    @property
    def eta_centered_(self):
        eta = self.eta_
        return eta - eta.mean(axis=0, keepdims=True)

    # Return c_eff = gamma + eta_centered in window/source order.
    @property
    def c_eff_(self):
        return self.gamma_[None, :] + self.eta_centered_

    # Compute genetic-only or full predictions in participant blocks.
    def _predict(self, data, *, genetic_only):
        self._require_fitted()

        if genetic_only:
            for name in ("source_names", "window_names"):
                if getattr(data, name) != getattr(self, name + "_"):
                    raise ValueError(f"{name} or their order differ from the fitted model.")

            # s_i = sum_wk(S_iwk * c_eff_wk); no intercept or covariate terms.
            values = np.empty(data.n_samples, dtype=np.float64)
            c_eff = self.c_eff_

            for start in range(0, data.n_samples, self.block_size):
                stop = min(start + self.block_size, data.n_samples)
                S_window = self.scaler_.transform_windows(data.window_scores[start:stop])

                # Sum windows and sources, keeping one value per participant.
                values[start:stop] = np.einsum("iwk,wk->i", S_window, c_eff, dtype=np.float64)

            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite genetic predictions.")
            return values

        # Manuscript eq:blocprs: ell = alpha_0 + C @ alpha + genetic score.
        design = Design(data, self.scaler_, block_size=self.block_size)
        values = np.empty(design.n_samples, dtype=np.float64)
        cursor = 0

        for X, _ in design.iter_blocks(include_outcome=False):
            end = cursor + len(X)
            values[cursor:end] = X @ self.coef_
            cursor = end

        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite full predictions.")
        return values

    # Compute the genetic score without covariates, intercept or sigmoid.
    def predict_prs(self, data):
        return self._predict(data, genetic_only=True)

    # Compute full ell: intercept + covariates + genetics, in model units.
    def predict_linear(self, data):
        return self._predict(data, genetic_only=False)

    # Convert the full binary predictor to P(y=1) for each participant.
    def predict_proba(self, data):
        if self.family != "binomial":
            raise ValueError("predict_proba requires a binomial model.")
        return expit(self.predict_linear(data))

    # Evaluate covariate-adjusted R^2; only the evaluation regressions are refitted.
    def score(self, data):
        self._require_fitted()
        if data.y is None:
            raise ValueError("Evaluation requires outcomes.")
        if data.covariate_names != self.covariate_names_:
            raise ValueError("Evaluation requires the fitted covariate names and order.")

        metric = NagelkerkeR2 if self.family == "binomial" else IncrementalR2
        return metric(data.y, data.covariates).score(self.predict_prs(data))

    # Save coefficients, training means/SDs and settings in one NumPy archive.
    def save(self, path):
        self._require_fitted()
        path = Path(path)
        metadata = {
            "format": "blocprs-model-v1",
            "family": self.family,
            "source_names": self.source_names_,
            "window_names": self.window_names_,
            "covariate_names": self.covariate_names_,
            "selected_lambda": self.selected_lambda_,
            "loss_reduction": "mean",
            "centering": "uniform",
            "scaling_scope": "outer_training",
            "cv_metric": "nagelkerke_r2_logit_v1" if self.family == "binomial" else "incremental_r2_ols_v1",
            "cv_results": self.cv_results_,
            "fit_info": self.fit_info_,
            "n_training_samples": self.n_training_samples_,
            "parameters": {
                "lambda_": None if self.lambda_ is None else float(self.lambda_),
                "lambda_grid": None if self.lambda_grid is None else [float(value) for value in self.lambda_grid],
                "cv_folds": int(self.cv_folds),
                "cv_max_samples": None if self.cv_max_samples is None else int(self.cv_max_samples),
                "random_state": None if self.random_state is None else int(self.random_state),
                "block_size": int(self.block_size),
            },
        }
        scaler = self.scaler_

        # Replace the destination only after the archive has been written.
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:

                # Keep v1 archive keys unchanged; only the in-code field names differ.
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
                                    coef=self.coef_, window_mean=scaler.window_mean,
                                    source_scale=scaler.source_sd,
                                    covariate_mean=scaler.cov_mean, covariate_scale=scaler.cov_sd,
                                    outcome_mean=np.asarray(scaler.y_mean), outcome_scale=np.asarray(scaler.y_sd))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    # Restore a fitted model from this package's archive, not a legacy pickle.
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if (metadata.get("format") != "blocprs-model-v1"
                or metadata.get("loss_reduction") != "mean"
                or metadata.get("centering") != "uniform"
                or metadata.get("scaling_scope") != "outer_training"):
                raise ValueError("Unsupported model format or formulation.")

            # Translate v1 archive keys to the explicit mean/SD field names.
            archive_names = {
                "window_mean": "window_mean",
                "source_scale": "source_sd",
                "covariate_mean": "cov_mean",
                "covariate_scale": "cov_sd",
                "outcome_mean": "y_mean",
                "outcome_scale": "y_sd",
            }

            # Reject nonnumeric arrays before Scaler performs any conversion.
            arrays = {}
            for stored_name, name in archive_names.items():
                value = saved[stored_name].copy()
                if value.dtype.kind not in "iuf":
                    raise ValueError(f"Saved {name} must be a real numeric array.")
                arrays[name] = value

            # Check the restored source/window/covariate order, means and SDs.
            scaler = Scaler(family=metadata["family"], source_names=metadata["source_names"],
                            window_names=metadata["window_names"], covariate_names=metadata["covariate_names"],
                            **arrays)
            n_sources = len(scaler.source_names)
            n_windows = len(scaler.window_names)
            n_covariates = len(scaler.covariate_names)
            n_coefficients = 1 + n_covariates + n_sources * (1 + n_windows)

            coef = saved["coef"].copy()
            if coef.dtype.kind not in "iuf" or coef.shape != (n_coefficients,) or not np.isfinite(coef).all():
                raise ValueError("Invalid saved coefficients.")

            selected_lambda = float(metadata["selected_lambda"])
            if not np.isfinite(selected_lambda) or selected_lambda < 0:
                raise ValueError("Invalid saved penalty.")

            model = cls(scaler.family, **metadata["parameters"])
            if model.block_size < 1:
                raise ValueError("Invalid saved block_size.")

            model.scaler_ = scaler
            model.source_names_ = scaler.source_names
            model.window_names_ = scaler.window_names
            model.covariate_names_ = scaler.covariate_names
            model.coef_ = coef
            model.selected_lambda_ = selected_lambda
            model.cv_results_ = metadata["cv_results"]
            model.fit_info_ = metadata["fit_info"]
            model.n_training_samples_ = int(metadata["n_training_samples"])

        return model
