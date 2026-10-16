"""Prepare windowed PRS data for model fitting and prediction.

Window scores have shape (participants, windows, sources).
Data are processed in participant blocks to limit RAM use.
Inputs must already be harmonized and in the same participant order.
Training means and SDs are reused for validation and prediction.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


# Check that names are unique and nonempty, keeping their supplied order.
def _names(values, size, label):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of names, not one string.")
    values = tuple(values)
    if (len(values) != size
        or any(not isinstance(v, str) or not v.strip() or v != v.strip()
               or any(c in v for c in "\t\n\r") for v in values)
        or len(set(values)) != size):
        raise ValueError(f"{label} must contain {size} unique, nonempty strings in array order.")
    return values


# Check numeric type and shape without copying existing score arrays.
def _numeric(values, ndim, label):
    array = np.asanyarray(values)
    if np.ma.isMaskedArray(array) or array.ndim != ndim or array.dtype.kind not in "biuf":
        raise ValueError(f"{label} must be a {ndim}-dimensional ordinary real numeric array.")
    return array


# Check that the number of participants per block is a positive integer.
def _block_size(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("block_size must be a positive integer.")
    return int(value)


# Copy values as float32 and reject missing, infinite or out-of-range values.
def _float32(values, label):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite and representable as float32.")
    return result


class PRSData:
    """Store window scores, optional phenotypes, covariates and participant IDs.

    Scores have shape (participants, windows, sources); covariates have shape
    (participants, covariates), and y has one value per participant.
    Inputs must already be harmonized and in matching participant order.
    Names retain the supplied order. IDs are checked, not used to align arrays.
    Existing arrays are used directly; do not change them during fitting or
    prediction. Score values are checked as each participant block is processed.
    """

    # Record aligned raw arrays and validate layout and participant IDs.
    def __init__(self, window_scores, source_names, window_names, *, y=None,
                 covariates=None, covariate_names=None, sample_ids=None):

        # Keep scores and their source/window names in the supplied order.
        self.window_scores = _numeric(window_scores, 3, "window_scores")
        self.n_samples, self.n_windows, self.n_sources = self.window_scores.shape
        if min(self.window_scores.shape) < 1:
            raise ValueError("window_scores must contain participants, windows, and sources.")
        self.source_names = _names(source_names, self.n_sources, "source_names")
        self.window_names = _names(window_names, self.n_windows, "window_names")

        self.y = None if y is None else _numeric(y, 1, "y")
        if self.y is not None and self.y.shape != (self.n_samples,):
            raise ValueError("y must contain one outcome per participant.")

        self.covariates = (np.empty((self.n_samples, 0), dtype=np.float32) if covariates is None
                           else _numeric(covariates, 2, "covariates"))
        if self.covariates.shape[0] != self.n_samples:
            raise ValueError("covariates must contain one row per participant.")
        self.n_covariates = self.covariates.shape[1]
        if covariate_names is None:
            covariate_names = tuple(f"covariate_{j}" for j in range(self.n_covariates))
        self.covariate_names = _names(covariate_names, self.n_covariates, "covariate_names")

        # Check ID values and duplicates without joining or reordering arrays.
        self.sample_ids = None
        if sample_ids is not None:
            ids = np.asanyarray(sample_ids)
            if (ids.ndim not in (1, 2) or ids.shape[0] != self.n_samples
                or (ids.ndim == 2 and ids.shape[1] == 0)):
                raise ValueError("sample_ids must have N rows and at least one ID field.")

            seen = set()
            for row in ids.reshape(self.n_samples, -1):
                key = tuple(row.tolist())
                for value in key:
                    if isinstance(value, (str, bytes)):
                        valid = bool(value.strip())
                    elif isinstance(value, (int, float, bool, np.number)):
                        valid = bool(np.isfinite(value))
                    else:
                        valid = False
                    if not valid:
                        raise ValueError("sample_ids contain a missing or unsupported ID value.")
                if key in seen:
                    raise ValueError("sample_ids contain duplicate participant IDs.")
                seen.add(key)
            self.sample_ids = ids


@dataclass
class Scaler:
    """Store training means and SDs for standardizing model inputs.

    window_mean: W x K; source_sd: K; cov_mean/cov_sd: C; y_mean/y_sd: scalar.
    W, K and C are the numbers of windows, sources and covariates.
    Each source SD comes from its genome-wide sum of window scores (ddof=0).
    Stored values use float32. Constant columns use 1 in place of a zero SD.
    Covariates and phenotypes with at most two distinct values stay unchanged,
    using mean=0 and SD=1. Prediction reuses these training values.
    """

    family: str
    source_names: tuple
    window_names: tuple
    covariate_names: tuple
    window_mean: np.ndarray
    source_sd: np.ndarray
    cov_mean: np.ndarray
    cov_sd: np.ndarray
    y_mean: float
    y_sd: float

    # Check the training means and SDs, then store them as float32.
    def __post_init__(self):
        if self.family not in ("gaussian", "binomial"):
            raise ValueError("family must be 'gaussian' or 'binomial'.")

        self.source_names = _names(self.source_names, len(self.source_names), "source_names")
        self.window_names = _names(self.window_names, len(self.window_names), "window_names")
        self.covariate_names = _names(self.covariate_names, len(self.covariate_names), "covariate_names")
        if not self.source_names or not self.window_names:
            raise ValueError("Scaler requires at least one source and window.")

        shapes = {
            "window_mean": (len(self.window_names), len(self.source_names)),
            "source_sd": (len(self.source_names),),
            "cov_mean": (len(self.covariate_names),),
            "cov_sd": (len(self.covariate_names),),
        }
        for name, shape in shapes.items():
            value = _float32(getattr(self, name), name)
            if value.shape != shape or (name in ("source_sd", "cov_sd") and np.any(value <= 0)):
                raise ValueError(f"Invalid {name}; expected shape {shape} and positive SD denominators.")
            setattr(self, name, value)

        for name in ("y_mean", "y_sd"):
            value = _float32(getattr(self, name), name)
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar.")
            setattr(self, name, float(value))

        if self.y_sd <= 0:
            raise ValueError("y_sd must be positive.")
        if self.family == "binomial" and (self.y_mean != 0 or self.y_sd != 1):
            raise ValueError("Binomial outcomes must remain unscaled.")

    # Compute training means and SDs in blocks, accumulating sums in float64.
    @classmethod
    def fit(cls, data, family, block_size=8192):
        block_size = _block_size(block_size)
        if family not in ("gaussian", "binomial"):
            raise ValueError("family must be 'gaussian' or 'binomial'.")
        if data.y is None:
            raise ValueError("Fitting a scaler requires training outcomes.")

        window_sum = np.zeros((data.n_windows, data.n_sources), dtype=np.float64)
        global_sum = np.zeros(data.n_sources, dtype=np.float64)
        cov_sum = np.zeros(data.n_covariates, dtype=np.float64)
        global_ss = np.zeros_like(global_sum)
        cov_ss = np.zeros_like(cov_sum)
        y_sum = y_ss = 0.0
        distinct = [set() for _ in range(data.n_covariates + 1)]

        # Compute means first, then squared deviations for more accurate SDs.
        for pass_number in (0, 1):
            for start in range(0, data.n_samples, block_size):
                rows = slice(start, min(start + block_size, data.n_samples))
                windows = _float32(data.window_scores[rows], "window_scores")
                covariates = _float32(data.covariates[rows], "covariates")
                y = _float32(data.y[rows], "y")
                if family == "binomial" and not np.isin(data.y[rows], (0, 1)).all():
                    raise ValueError("Binomial outcomes must be exactly 0 or 1.")

                # Sum windows to get each source's genome-wide score.
                global_score = windows.sum(axis=1, dtype=np.float64)
                if pass_number == 0:
                    window_sum += windows.sum(axis=0, dtype=np.float64)
                    global_sum += global_score.sum(axis=0, dtype=np.float64)
                    cov_sum += covariates.sum(axis=0, dtype=np.float64)
                    y_sum += float(y.sum(dtype=np.float64))

                # Identify columns with at most two distinct values before rounding.
                    for j in range(data.n_covariates + 1):
                        if len(distinct[j]) <= 2:
                            raw = data.y[rows] if j == data.n_covariates else data.covariates[rows, j]
                            distinct[j].update(np.unique(raw)[:3].tolist())

                else:
                    global_score -= global_mean
                    global_ss += np.einsum("ij,ij->j", global_score, global_score)
                    cov_delta = covariates.astype(np.float64) - cov_mean
                    cov_ss += np.einsum("ij,ij->j", cov_delta, cov_delta)
                    y_delta = y.astype(np.float64) - y_mean
                    y_ss += float(y_delta @ y_delta)

            if pass_number == 0:
                global_mean = global_sum / data.n_samples
                cov_mean = cov_sum / data.n_samples
                y_mean = y_sum / data.n_samples

        if family == "binomial" and distinct[-1] != {0, 1}:
            raise ValueError("Binomial training outcomes must contain both classes.")

        # Population SD (ddof=0); use 1 for constant columns to avoid division by zero.
        source_sd = np.sqrt(global_ss / data.n_samples).astype(np.float32)
        cov_sd = np.sqrt(cov_ss / data.n_samples).astype(np.float32)
        source_sd[source_sd == 0] = 1
        cov_sd[cov_sd == 0] = 1

        # Leave covariates and phenotypes with at most two distinct values unchanged.
        for j in range(data.n_covariates):
            if len(distinct[j]) <= 2:
                cov_mean[j], cov_sd[j] = 0, 1
        y_sd = np.float32(np.sqrt(y_ss / data.n_samples))
        if family == "binomial" or len(distinct[-1]) <= 2:
            y_mean, y_sd = 0, 1
        elif y_sd == 0:
            y_sd = 1

        return cls(family, data.source_names, data.window_names, data.covariate_names,
                   window_sum / data.n_samples, source_sd, cov_mean, cov_sd, y_mean, y_sd)

    # Standardize windows using their training means and genome-wide source SDs.
    def transform_windows(self, values):
        result = _float32(values, "window_scores")
        if result.ndim != 3 or result.shape[1:] != self.window_mean.shape:
            raise ValueError("Window-score block does not match the fitted layout.")

        # S_window = (raw_window - training_window_mean) / training_source_SD.
        result -= self.window_mean
        result /= self.source_sd

        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized window scores.")
        return result

    # Standardize covariates using their training means and SDs.
    def transform_covariates(self, values):
        result = _float32(values, "covariates")
        if result.ndim != 2 or result.shape[1] != len(self.covariate_names):
            raise ValueError("Covariate block does not match the fitted layout.")

        result -= self.cov_mean
        result /= self.cov_sd

        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized covariates.")
        return result

    # Standardize y with training values; binary phenotypes remain unchanged.
    def transform_outcome(self, values):
        result = _float32(values, "y")
        if result.ndim != 1:
            raise ValueError("Outcome block must be a vector.")
        if self.family == "binomial" and not np.isin(values, (0, 1)).all():
            raise ValueError("Binomial outcomes must be exactly 0 or 1.")

        result -= np.float32(self.y_mean)
        result /= np.float32(self.y_sd)

        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized outcomes.")
        return result


class Design:
    """Build model inputs in participant blocks to limit RAM use.

    Each block contains an intercept, covariates, genome-wide source scores
    and centered window scores: [1, C, S_global, Z]. Training means and SDs
    stay fixed during validation and prediction. Selected participants retain
    their supplied order. Standardization uses float32; model blocks use float64.
    """

    # Check source/window/covariate order and select participants without copying scores.
    def __init__(self, data, scaler, rows=None, block_size=8192):
        self.data, self.scaler = data, scaler
        self.block_size = _block_size(block_size)
        for name in ("source_names", "window_names", "covariate_names"):
            if getattr(data, name) != getattr(scaler, name):
                raise ValueError(f"{name} or their order differ from the fitted scaler.")

        if rows is None:
            rows = slice(0, data.n_samples, 1)
        if isinstance(rows, slice):
            start, stop, step = rows.indices(data.n_samples)
            if step != 1:
                raise ValueError("Row slices must have step=1; use explicit indices otherwise.")
            self.rows = slice(start, stop, 1)
            self.n_samples = max(0, stop - start)
        else:
            rows = np.asarray(rows)
            if rows.dtype.kind == "b" and rows.shape == (data.n_samples,):
                rows = np.flatnonzero(rows)
            if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0)
                or np.any(rows >= data.n_samples) or len(np.unique(rows)) != len(rows)):
                raise ValueError("rows must be unique, in-range integer indices or an N-vector mask.")
            self.rows = rows.astype(np.int64, copy=True)
            self.n_samples = len(rows)

        if self.n_samples < 1:
            raise ValueError("Design must contain at least one participant.")

        self.n_covariates = data.n_covariates
        self.n_features = 1 + data.n_covariates + data.n_sources * (1 + data.n_windows)

    # Return each block's output positions and corresponding input participant rows.
    def _blocks(self):
        for start in range(0, self.n_samples, self.block_size):
            stop = min(start + self.block_size, self.n_samples)
            rows = (slice(self.rows.start + start, self.rows.start + stop)
                    if isinstance(self.rows, slice) else self.rows[start:stop])
            yield start, stop, rows

    # Build float64 (X, y) blocks in the selected participant order.
    # If y is absent or include_outcome=False, return y=None without reading phenotypes.
    def iter_blocks(self, *, include_outcome=True):
        data = self.data
        first_genetic = 1 + self.n_covariates
        first_window = first_genetic + data.n_sources

        for start, stop, rows in self._blocks():

            # X = [1, C, S_global, Z]; write window contrasts into X in place.
            X = np.empty((stop - start, self.n_features), dtype=np.float64)
            X[:, 0] = 1
            X[:, 1:first_genetic] = self.scaler.transform_covariates(data.covariates[rows])

            # S_global sums windows; Z subtracts each source's uniform window share.
            windows = X[:, first_window:].reshape(-1, data.n_windows, data.n_sources)
            windows[:] = self.scaler.transform_windows(data.window_scores[rows])
            global_score = X[:, first_genetic:first_window]
            global_score[:] = windows.sum(axis=1, dtype=np.float64)
            windows -= global_score[:, None, :] * (1.0 / data.n_windows)

            y = (None if not include_outcome or data.y is None
                 else self.scaler.transform_outcome(data.y[rows]).astype(np.float64))
            if not np.isfinite(X).all():
                raise ValueError("Nonfinite model design.")

            yield X, y

    # Collect transformed phenotypes without reading PRS; return None if absent.
    def outcome(self):
        if self.data.y is None:
            return None

        result = np.empty(self.n_samples, dtype=np.float64)
        for start, stop, rows in self._blocks():
            result[start:stop] = self.scaler.transform_outcome(self.data.y[rows])
        return result

    # Collect selected transformed N x C covariates without reading PRS.
    def covariates(self):
        result = np.empty((self.n_samples, self.n_covariates), dtype=np.float64)
        for start, stop, rows in self._blocks():
            result[start:stop] = self.scaler.transform_covariates(self.data.covariates[rows])
        return result
