"""Regional recalibration of a genome-wide MultiPRS mixture.

MultiPRS estimates a common set of source weights across the genome.
BLocPRS then estimates a global coefficient and regional adjustments
for the combined score. Both stages are fitted within each
cross-validation training fold.
"""

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.model_selection import KFold, StratifiedKFold

from .baseline_model import MultiPRS
from .cv import _genetic_predictions, _positive_integer, _subsample_indices
from .data import Design, PRSData, Scaler
from .metrics import IncrementalR2, NagelkerkeR2
from .model import BLocPRS
from .solvers import continuous_statistics, fit as solve


# Select matching participants without copying the full windowed source array.
def _rows(data, rows, scores=None, source_names=None):
    return PRSData(data.window_scores[rows] if scores is None else scores,
                   data.source_names if source_names is None else source_names,
                   data.window_names, y=None if data.y is None else data.y[rows],
                   covariates=data.covariates[rows], covariate_names=data.covariate_names,
                   sample_ids=None if data.sample_ids is None else data.sample_ids[rows])


# Check that the two independently prepared datasets describe the same rows.
def _aligned_inputs(windowed, genomewide):
    if not isinstance(windowed, PRSData) or not isinstance(genomewide, PRSData):
        raise ValueError("Supply windowed and genome-wide PRSData objects.")

    if windowed.y is None or genomewide.y is None or genomewide.n_windows != 1:
        raise ValueError("Both training inputs need outcomes; genome-wide scores need one window.")

    if windowed.source_names != genomewide.source_names or windowed.covariate_names != genomewide.covariate_names:
        raise ValueError("Source and covariate names must match in the same order.")

    if (windowed.sample_ids is None or genomewide.sample_ids is None
        or not np.array_equal(windowed.sample_ids, genomewide.sample_ids)):
        raise ValueError("Provide matching participant IDs in the same order for both training inputs.")

    if not np.array_equal(windowed.y, genomewide.y) or not np.array_equal(windowed.covariates, genomewide.covariates):
        raise ValueError("Phenotypes and covariates differ between the two training inputs.")


# Combine raw sources within each window using float32 arithmetic.
def _combined_data(data, source_weights, block_size, rows=None):
    n_samples = data.n_samples if rows is None else len(rows)
    combined = np.zeros((n_samples, data.n_windows, 1), dtype=np.float32)

    for start in range(0, n_samples, block_size):
        stop = min(start + block_size, n_samples)
        selected = slice(start, stop) if rows is None else rows[start:stop]

        for source, weight in enumerate(source_weights):
            values = np.array(data.window_scores[selected, :, source], dtype=np.float32, copy=True)
            values *= np.float32(weight)
            combined[start:stop, :, 0] += values

        if not np.isfinite(combined[start:stop]).all():
            raise ValueError("Nonfinite fixed-composition window scores.")

    return _rows(data, slice(None) if rows is None else rows, combined, ("MultiPRS",))


class FixedCompositionBLocPRS:
    """Two-stage PRS integration with a fixed genome-wide source composition.

    An unregularized MultiPRS fit defines the source mixture. A one-source
    BLocPRS model estimates its global coefficient and regional adjustments.
    Regional CV uses the BLocPRS penalty grid and evaluation metrics, with
    both stages refitted within each training fold. The final fit uses all
    training participants.

    Fitting constructs a combined float32 window-score matrix; prediction
    combines source scores in participant blocks.
    """

    # Set the regional penalty choices and participant-block size.
    def __init__(self, family="gaussian", *, lambda_=None, lambda_grid=None,
                 cv_folds=5, cv_max_samples=50000, random_state=42, block_size=8192):
        self.family = family
        self.lambda_ = lambda_
        self.lambda_grid = lambda_grid
        self.cv_folds = cv_folds
        self.cv_max_samples = cv_max_samples
        self.random_state = random_state
        self.block_size = block_size

    # Select the regional penalty using a new source mixture in every fold.
    def _select_lambda(self, data, genomewide_data, candidates):
        folds = _positive_integer(self.cv_folds, "cv_folds", minimum=2)
        available = data.n_samples

        if self.cv_max_samples is not None:
            available = min(available, _positive_integer(self.cv_max_samples, "cv_max_samples", minimum=folds))

        total = (available // folds) * folds

        if total < folds:
            raise ValueError("Too few training participants for the requested CV folds.")

        sampled = _subsample_indices(data.y, total, np.random.RandomState(self.random_state),
                                     stratify=self.family == "binomial", min_per_class=folds)
        sampled_y = data.y[sampled]
        splitter_type = StratifiedKFold if self.family == "binomial" else KFold
        splitter = splitter_type(n_splits=folds, shuffle=True, random_state=self.random_state)

        if self.family == "binomial":
            values, counts = np.unique(sampled_y, return_counts=True)

            if not np.array_equal(values, [0, 1]) or np.min(counts) < folds:
                raise ValueError("Binary CV needs at least cv_folds sampled participants in each class.")

        results = [{"lambda": float(value), "mean_score": None, "fold_scores": []} for value in candidates]
        fitting_order = sorted(results, key=lambda entry: -entry["lambda"]) if self.family == "binomial" else results

        for train_rows, validation_rows in splitter.split(np.zeros((total, 1)), sampled_y):

            # Refit the source mixture inside the fold so validation outcomes cannot set it.
            mixture = MultiPRS(self.family, variant="unregularized", block_size=self.block_size)
            mixture.fit(_rows(genomewide_data, sampled[train_rows]))

            # Convert standardized-score coefficients to weights on raw window scores.
            # The second-stage training centering accounts for the mixture's constant offset.
            source_weights = mixture.gamma_ / mixture.scaler_.source_sd
            combined = _combined_data(data, source_weights, self.block_size, sampled)
            training_data = _rows(combined, train_rows)
            validation_data = _rows(combined, validation_rows)
            del combined

            # Regional scaling is learned on fold-training rows, after mixing.
            scaler = Scaler.fit(training_data, self.family, self.block_size)
            training = Design(training_data, scaler, block_size=self.block_size)
            validation = Design(validation_data, scaler, block_size=self.block_size)
            metric_type = NagelkerkeR2 if self.family == "binomial" else IncrementalR2
            metric = metric_type(validation.outcome(), validation.covariates())
            statistics = continuous_statistics(training) if self.family == "gaussian" else None
            initial = None

            # Reuse cross-products or warm starts, using the shared BLocPRS solver.
            for entry in fitting_order:
                fitted = solve(training, self.family, entry["lambda"], 2 + training.n_covariates,
                               initial=initial, statistics=statistics)
                entry["fold_scores"].append(float(metric.score(_genetic_predictions(validation, fitted.coef))))

                if self.family == "binomial":
                    initial = fitted.coef

        for entry in results:
            entry["mean_score"] = float(np.mean(entry["fold_scores"]))

        # Ties keep the first supplied grid value.
        selected = max(results, key=lambda entry: entry["mean_score"])
        return selected["lambda"], results

    # Fit both stages on training participants and retain their fitted parameters.
    def fit(self, data, genomewide_data):
        _aligned_inputs(data, genomewide_data)
        self.block_size = _positive_integer(self.block_size, "block_size")

        if self.family not in ("gaussian", "binomial"):
            raise ValueError("family must be 'gaussian' or 'binomial'.")

        candidates = np.logspace(-3, 3, 25) if self.lambda_grid is None else np.asarray(self.lambda_grid, dtype=float)

        if (candidates.ndim != 1 or len(candidates) == 0 or not np.isfinite(candidates).all()
            or np.any(candidates < 0) or len(np.unique(candidates)) != len(candidates)):
            raise ValueError("lambda_grid must contain distinct finite nonnegative values.")

        if self.lambda_ is None:
            selected, results = self._select_lambda(data, genomewide_data, candidates)
        else:

            if self.lambda_grid is not None or not np.isfinite(self.lambda_) or self.lambda_ < 0:
                raise ValueError("Choose one finite nonnegative lambda_ or a lambda_grid, not both.")

            selected, results = float(self.lambda_), []

        mixture = MultiPRS(self.family, variant="unregularized", block_size=self.block_size).fit(genomewide_data)

        # Freeze one source composition across windows; only its regional scaling can vary.
        source_weights = mixture.gamma_ / mixture.scaler_.source_sd
        combined = _combined_data(data, source_weights, self.block_size)
        regional = BLocPRS(self.family, lambda_=selected, block_size=self.block_size).fit(combined)

        # Keep an existing fitted model intact if either new fit fails.
        self.mixture_, self.source_weights_, self.regional_ = mixture, source_weights, regional
        self.source_names_ = data.source_names
        self.window_names_ = data.window_names
        self.covariate_names_ = data.covariate_names
        self.selected_lambda_ = float(selected)
        self.cv_results_ = results
        self.n_training_samples_ = data.n_samples
        self.fit_info_ = {"mixture": self.mixture_.fit_info_, "regional": self.regional_.fit_info_}
        return self

    # Check the original source and window order before applying the fitted mixture.
    def _check_prediction_data(self, data):
        if not hasattr(self, "regional_"):
            raise ValueError("Fit or load a fixed-composition model before prediction.")

        if data.source_names != self.source_names_ or data.window_names != self.window_names_:
            raise ValueError("Source/window names or their order differ from the fitted model.")

    # Combine and predict small participant blocks, keeping both fitted stages frozen.
    def _predict(self, data, genetic_only):
        self._check_prediction_data(data)
        values = np.empty(data.n_samples, dtype=np.float64)
        predict = self.regional_.predict_prs if genetic_only else self.regional_.predict_linear

        for start in range(0, data.n_samples, self.block_size):
            stop = min(start + self.block_size, data.n_samples)
            block = _rows(data, slice(start, stop))
            values[start:stop] = predict(_combined_data(block, self.source_weights_, self.block_size))

        return values

    # Return genetics only, with no covariate effects, intercept or sigmoid.
    def predict_prs(self, data):
        return self._predict(data, genetic_only=True)

    # Return the regional model's full fitted linear predictor.
    def predict_linear(self, data):
        return self._predict(data, genetic_only=False)

    # Convert the full binary predictor to a case probability.
    def predict_proba(self, data):
        if self.family != "binomial":
            raise ValueError("predict_proba requires a binomial model.")

        return expit(self.predict_linear(data))

    # Label the restricted comparator separately from full BLocPRS.
    @property
    def method_name_(self):
        return "Fixed-composition regional MultiPRS"

    # Evaluate the frozen genetic score using the common covariate-adjusted metric.
    def score(self, data):
        self._check_prediction_data(data)

        if data.y is None or data.covariate_names != self.covariate_names_:
            raise ValueError("Evaluation requires outcomes and the fitted covariate names/order.")

        metric = NagelkerkeR2 if self.family == "binomial" else IncrementalR2
        return metric(data.y, data.covariates).score(self.predict_prs(data))

    # Store both existing model archives inside one pickle-free NumPy archive.
    def save(self, path):
        if not hasattr(self, "regional_"):
            raise ValueError("Fit or load a fixed-composition model before saving it.")

        path = Path(path)
        metadata = {
            "format": "blocprs-fixed-composition-v1", "family": self.family,
            "source_names": self.source_names_, "window_names": self.window_names_,
            "covariate_names": self.covariate_names_, "cv_results": self.cv_results_,
            "n_training_samples": self.n_training_samples_,
            "cv_scaling_scope": "fold_training", "mixture_refit_per_fold": True,
            "cv_metric": "nagelkerke_r2_logit_v1" if self.family == "binomial" else "incremental_r2_ols_v1",
            "parameters": {"lambda_": None if self.lambda_ is None else float(self.lambda_),
                           "lambda_grid": None if self.lambda_grid is None else [float(value) for value in self.lambda_grid],
                           "cv_folds": int(self.cv_folds),
                           "cv_max_samples": None if self.cv_max_samples is None else int(self.cv_max_samples),
                           "random_state": None if self.random_state is None else int(self.random_state),
                           "block_size": int(self.block_size)},
        }

        with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as temporary:
            temporary = Path(temporary)
            self.mixture_.save(temporary / "mixture.npz")
            self.regional_.save(temporary / "regional.npz")
            mixture = np.frombuffer((temporary / "mixture.npz").read_bytes(), dtype=np.uint8)
            regional = np.frombuffer((temporary / "regional.npz").read_bytes(), dtype=np.uint8)
            np.savez_compressed(temporary / "model.npz", metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
                                mixture=mixture, regional=regional)
            os.replace(temporary / "model.npz", path)

    # Restore both stages with their original archive loaders and frozen scaling.
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))

            if (metadata.get("format") != "blocprs-fixed-composition-v1"
                or metadata.get("cv_scaling_scope") != "fold_training"
                or metadata.get("mixture_refit_per_fold") is not True):
                raise ValueError("Unsupported fixed-composition model format.")

            model = cls(metadata["family"], **metadata["parameters"])
            model.mixture_ = MultiPRS.load(io.BytesIO(saved["mixture"].tobytes()))
            model.regional_ = BLocPRS.load(io.BytesIO(saved["regional"].tobytes()))

        model.source_names_ = tuple(metadata["source_names"])
        model.window_names_ = tuple(metadata["window_names"])
        model.covariate_names_ = tuple(metadata["covariate_names"])

        if (model.mixture_.family != model.family or model.regional_.family != model.family
            or model.mixture_.source_names_ != model.source_names_
            or model.regional_.source_names_ != ("MultiPRS",)
            or model.regional_.window_names_ != model.window_names_
            or model.mixture_.covariate_names_ != model.covariate_names_
            or model.regional_.covariate_names_ != model.covariate_names_):
            raise ValueError("Stored mixture and regional model labels do not match.")

        model.source_weights_ = model.mixture_.gamma_ / model.mixture_.scaler_.source_sd
        model.selected_lambda_ = model.regional_.selected_lambda_
        model.cv_results_ = metadata["cv_results"]
        model.n_training_samples_ = int(metadata["n_training_samples"])
        model.fit_info_ = {"mixture": model.mixture_.fit_info_, "regional": model.regional_.fit_info_}
        return model
