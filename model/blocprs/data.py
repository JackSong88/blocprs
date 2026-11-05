"""Window-level polygenic scores and training-derived standardization.

Scores are organized by participant, genomic window and source.
Blockwise processing improves memory efficiency during fitting and prediction.
"""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from participant_utils import id_positions


class _ScoreView:
    """Read selected participants and sources one requested block at a time."""

    # Stores index mappings without copying the score matrix.
    def __init__(self, scores, rows=None, sources=None):
        rows = np.arange(scores.shape[0]) if rows is None else np.asarray(rows)
        sources = np.arange(scores.shape[2]) if sources is None else np.asarray(sources)

        if isinstance(scores, _ScoreView):
            rows, sources = scores.rows[rows], scores.sources[sources]
            scores = scores.scores

        self.scores, self.rows, self.sources = scores, rows, sources
        self.shape = (len(rows), scores.shape[1], len(sources))
        self.ndim, self.dtype = 3, scores.dtype

    # Reads only the requested rows, windows and sources into memory.
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Finds ... in the requested indexing, if present.
        ellipsis = None

        for position, item in enumerate(key):
            if item is Ellipsis:
                ellipsis = position
                break

        if ellipsis is not None:
            before = key[:ellipsis]
            after = key[ellipsis + 1:]
            missing_dimensions = 3 - len(before) - len(after)

            # slice(None) is :, selecting everything along that dimension.
            full_selections = (slice(None),) * missing_dimensions
            key = before + full_selections + after

        if len(key) > 3:
            raise IndexError("Window scores have three dimensions.")

        # Unspecified trailing dimensions also select all entries: view[rows] means view[rows, :, :].
        missing_dimensions = 3 - len(key)
        key = key + (slice(None),) * missing_dimensions

        for index in key[1:]:
            if not isinstance(index, (slice, Integral)) or isinstance(index, (bool, np.bool_)):
                raise IndexError("Use scalar/slice window and source indices; select_sources handles named source lists.")

        # Maps the requested participants and sources back to the original score array.
        row_indices = self.rows[key[0]]
        window_indices = np.arange(self.shape[1])[key[1]]
        source_indices = self.sources[key[2]]

        # Converts a single index to a one-element array for np.ix_.
        rows = np.atleast_1d(row_indices)
        windows = np.atleast_1d(window_indices)
        sources = np.atleast_1d(source_indices)

        # np.ix_ selects every requested row/window/source combination, not paired indices.
        values = self.scores[np.ix_(rows, windows, sources)]
        scalar_axes = []

        # Integer indexing drops that dimension, while a slice keeps it.
        for axis, indices in enumerate((row_indices, window_indices, source_indices)):
            if np.ndim(indices) == 0:
                scalar_axes.append(axis)

        return values.squeeze(axis=tuple(scalar_axes))


# -- Sanity Checks used throughout --
# Checks that names are unique and nonempty, keeping their supplied order.
def _names(values, size, label):
    assert not isinstance(values, (str, bytes))

    values = tuple(values)

    if (len(values) != size
        or any(not isinstance(v, str) or not v.strip() or v != v.strip()
               or any(c in v for c in "\t\n\r") for v in values)
        or len(set(values)) != size):
        raise ValueError(f"{label} must contain {size} unique, nonempty strings in array order.")

    return values


# Checks numeric type and shape without copying existing score arrays.
def _numeric(values, ndim, label):
    array = values if isinstance(values, _ScoreView) else np.asanyarray(values)

    assert (not np.ma.isMaskedArray(array) and array.ndim == ndim and array.dtype.kind in "biuf"), f"Invalid {label} type or shape."

    return array


# Checks that the number of participants per block is a positive integer.
def _block_size(value):
    assert not isinstance(value, bool) and isinstance(value, Integral) and value >= 1

    return int(value)


# Copies values as float32 and rejects missing, infinite or out-of-range values.
def _float32(values, label):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32, copy=True)

    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite and representable as float32.")

    return result


class PRSData:
    """Aligned window-level PRSs, phenotypes, covariates and participant IDs.

    Raw scores S_tilde_iwk have shape (participants, windows, sources); covariates have shape
    (participants, covariates). Optional y contains one phenotype per participant.
    Inputs must already be harmonized and in matching participant order.
    Names retain the supplied order. IDs are checked, not used to align arrays.
    Score values are checked as each participant block is processed.
    """

    # Records aligned raw arrays and checks layout and participant IDs.
    def __init__(self, window_scores, source_names, window_names, *, y=None,
                 covariates=None, covariate_names=None, sample_ids=None):

        # Keeps scores and their source/window names in the supplied order.
        self.window_scores = _numeric(window_scores, 3, "window_scores")
        self.n_samples, self.n_windows, self.n_sources = self.window_scores.shape

        assert min(self.window_scores.shape) >= 1

        self.source_names = _names(source_names, self.n_sources, "source_names")
        self.window_names = _names(window_names, self.n_windows, "window_names")

        self.y = None if y is None else _numeric(y, 1, "y")

        assert self.y is None or self.y.shape == (self.n_samples,)

        if covariates is None:
            self.covariates = np.empty((self.n_samples, 0), dtype=np.float32)
        else:
            self.covariates = _numeric(covariates, 2, "covariates")

        assert self.covariates.shape[0] == self.n_samples

        self.n_covariates = self.covariates.shape[1]

        if covariate_names is None:
            covariate_names = tuple(f"covariate_{j}" for j in range(self.n_covariates))

        self.covariate_names = _names(covariate_names, self.n_covariates, "covariate_names")

        # Checks ID values and duplicates without joining or reordering arrays.
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


# Loads one saved split, keeping large score arrays on disk.
def load_dataset(path, split="train", ids_file=None):
    """Load a dataset as PRSData, without standardizing its values.

    Select the stored 'train', 'test' or 'all' split. Only split=None combines
    all stored rows; 'all' is the transfer split, not a train/test combination.
    An ID file selects participants within that split, retaining file order.
    """
    path = Path(path)
    participants = pd.read_csv(path / "participants.tsv", sep="\t", dtype={"FID": str, "IID": str},
                               float_precision="round_trip")
    ids = participants[["FID", "IID"]]

    # Checks missing and duplicate IDs across all stored splits.
    if ids.isna().any().any() or any(ids[column].str.strip().eq("").any() for column in ids):
        raise ValueError("Saved participants contain missing participant IDs.")

    if ids.duplicated().any():
        raise ValueError("Saved participants contain duplicate FID/IID pairs.")

    # Reads the names that define the score and covariate array columns.
    features = pd.read_csv(path / "features.tsv", sep="\t", dtype=str)
    covariate_names = pd.read_csv(path / "covariates.tsv", sep="\t", dtype=str)["covariate"].tolist()
    windows = features["window"].drop_duplicates().tolist()
    sources = features["source"].drop_duplicates().tolist()

    # Checks that features match the saved array's window-major order.
    if list(zip(features["window"], features["source"])) != [(window, source) for window in windows for source in sources]:
        raise ValueError("Saved feature names do not describe a complete window-major grid.")

    # Opens the saved arrays without loading the full score matrix into RAM.
    scores = np.load(path / "scores.npy", mmap_mode="r", allow_pickle=False)
    covariates = np.load(path / "covariates.npy", mmap_mode="r", allow_pickle=False)

    if scores.shape != (len(participants), len(windows), len(sources)) or covariates.shape != (len(participants), len(covariate_names)):
        raise ValueError("Saved arrays and their participant/feature names have inconsistent dimensions.")

    # Selects the saved split first; an ID file can only narrow that selection.
    if split is None:
        rows = np.arange(len(participants))
    else:
        rows = np.flatnonzero(participants["split"].eq(split))

    if ids_file is not None:
        rows = rows[id_positions(participants.iloc[rows], ids_file)]

    if len(rows) == 0:
        raise ValueError(f"Split {split!r} contains no selected participants.")

    # Contiguous splits remain ordinary slices; other selections are blockwise views.
    contiguous = np.all(np.diff(rows) == 1)

    if contiguous:
        selection = slice(int(rows[0]), int(rows[-1]) + 1)
        selected_scores = scores[selection]
    else:
        selection = rows
        selected_scores = _ScoreView(scores, rows)

    selected = participants.iloc[selection]

    return PRSData(selected_scores, sources, windows, y=selected["phenotype"].to_numpy(dtype=float),
                   covariates=covariates[selection], covariate_names=covariate_names,
                   sample_ids=selected[["FID", "IID"]].to_numpy())


# Selects named sources in the supplied order, without copying the score tensor.
def select_sources(data, source_names):
    source_names = tuple(source_names)

    assert source_names and len(set(source_names)) == len(source_names)

    if source_names == data.source_names:
        return data

    sources = [data.source_names.index(name) for name in source_names]

    return PRSData(_ScoreView(data.window_scores, sources=sources), source_names, data.window_names,
                   y=data.y, covariates=data.covariates, covariate_names=data.covariate_names,
                   sample_ids=data.sample_ids)


@dataclass
class GenomewideScaler:
    """Training mu_k and sigma_k of raw S_tilde_ik; covariates and y remain unscaled."""

    source_mean: np.ndarray
    source_sd: np.ndarray

    # Computes mu_k and population SD sigma_k across the supplied training participants.
    @classmethod
    def fit(cls, sources):
        sd = np.std(sources, axis=0, ddof=0)
        sd[sd == 0] = 1
        return cls(np.mean(sources, axis=0), sd)

    # Computes S_ik = (S_tilde_ik - mu_k) / sigma_k using saved training values.
    def transform(self, sources):
        return (sources - self.source_mean) / self.source_sd


@dataclass
class Scaler:
    """Training means and standard deviations for model standardization.

    window_mean stores mu_wk (W x K); source_sd stores sigma_k (K).
    cov_mean/cov_sd have one value per covariate; y_mean/y_sd are scalars.
    W, K and C are the numbers of windows, sources and covariates.
    mu_wk is the training mean of raw S_tilde_iwk. sigma_k is the training SD
    of S_tilde_ik = sum_w S_tilde_iwk (ddof=0), not a separate SD per window.
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

    # Checks the training means and SDs, then stores them as float32.
    def __post_init__(self):
        assert self.family in ("gaussian", "binomial")

        self.source_names = _names(self.source_names, len(self.source_names), "source_names")
        self.window_names = _names(self.window_names, len(self.window_names), "window_names")
        self.covariate_names = _names(self.covariate_names, len(self.covariate_names), "covariate_names")

        assert self.source_names and self.window_names

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

            assert value.ndim == 0

            setattr(self, name, float(value))

        assert self.y_sd > 0

        if self.family == "binomial" and (self.y_mean != 0 or self.y_sd != 1):
            raise ValueError("Binomial outcomes must remain unscaled.")

    # Computes training mu_wk and sigma_k in blocks of individuals, with float64 sums.
    @classmethod
    def fit(cls, data, family, block_size=8192, *, rows=None):
        block_size = _block_size(block_size)

        assert family in ("gaussian", "binomial")

        assert data.y is not None

        # Selects fold-training rows without copying the full score matrix.
        if rows is not None:
            rows = np.asarray(rows)

            if (rows.ndim != 1 or not pd.api.types.is_integer_dtype(rows.dtype) or rows.size == 0
                or np.any(rows < 0) or np.any(rows >= data.n_samples)
                or len(np.unique(rows)) != len(rows)):
                raise ValueError("rows must be unique, in-range integer indices.")

        # Keeps running totals for mu_wk, mu_k and sigma_k, plus covariates and y.
        n_samples = data.n_samples if rows is None else len(rows)
        window_sum = np.zeros((data.n_windows, data.n_sources), dtype=np.float64)
        global_sum = np.zeros(data.n_sources, dtype=np.float64)
        cov_sum = np.zeros(data.n_covariates, dtype=np.float64)
        global_ss = np.zeros_like(global_sum)
        cov_ss = np.zeros_like(cov_sum)
        y_sum = y_ss = 0.0
        distinct = [set() for _ in range(data.n_covariates + 1)]

        # Pass 1 accumulates window totals for mu_wk and computes genome-wide mu_k.
        # Pass 2 rereads all training blocks to compute sigma_k around the known mu_k.
        for pass_number in (0, 1):
            for start in range(0, n_samples, block_size):
                stop = min(start + block_size, n_samples)
                selection = slice(start, stop) if rows is None else rows[start:stop]
                windows = _float32(data.window_scores[selection], "window_scores")
                covariates = _float32(data.covariates[selection], "covariates")
                y = _float32(data.y[selection], "y")

                # S_tilde_ik = sum_w S_tilde_iwk, separately for each participant and source.
                global_score = windows.sum(axis=1, dtype=np.float64)

                if pass_number == 0:

                    # Sums over participants; dividing these totals by N gives the means.
                    window_sum += windows.sum(axis=0, dtype=np.float64)
                    global_sum += global_score.sum(axis=0, dtype=np.float64)
                    cov_sum += covariates.sum(axis=0, dtype=np.float64)
                    y_sum += float(y.sum(dtype=np.float64))

                    # Identifies columns with at most two distinct values before rounding.
                    for j in range(data.n_covariates + 1):
                        if len(distinct[j]) <= 2:
                            if j == data.n_covariates:
                                raw = data.y[selection]
                            else:
                                raw = data.covariates[selection, j]

                            distinct[j].update(np.unique(raw)[:3].tolist())

                else:
                    global_score -= global_mean

                    # Accumulates sum_i (S_tilde_ik - mu_k)^2, one total per source.
                    global_ss += np.einsum("ij,ij->j", global_score, global_score)
                    cov_delta = covariates.astype(np.float64) - cov_mean
                    cov_ss += np.einsum("ij,ij->j", cov_delta, cov_delta)
                    y_delta = y.astype(np.float64) - y_mean
                    y_ss += float(y_delta @ y_delta)

            if pass_number == 0:

                # Checks binary labels once, using the original values before float32 rounding.
                if family == "binomial" and distinct[-1] != {0, 1}:
                    raise ValueError("Binomial training outcomes must contain both 0 and 1 and no other values.")

                # mu_k = sum_i S_tilde_ik / N, using every selected training participant.
                global_mean = global_sum / n_samples
                cov_mean = cov_sum / n_samples
                y_mean = y_sum / n_samples

        # sigma_k = sqrt(sum_i (S_tilde_ik - mu_k)^2 / N), using population SDs.
        # Constant columns use a denominator of 1 to avoid division by zero.
        source_sd = np.sqrt(global_ss / n_samples).astype(np.float32)
        cov_sd = np.sqrt(cov_ss / n_samples).astype(np.float32)
        source_sd[source_sd == 0] = 1
        cov_sd[cov_sd == 0] = 1

        # Leaves covariates and phenotypes with at most two distinct values unchanged.
        for j in range(data.n_covariates):
            if len(distinct[j]) <= 2:
                cov_mean[j], cov_sd[j] = 0, 1

        y_sd = np.float32(np.sqrt(y_ss / n_samples))

        if family == "binomial" or len(distinct[-1]) <= 2:
            y_mean, y_sd = 0, 1
        elif y_sd == 0:
            y_sd = 1

        # mu_wk = sum_i S_tilde_iwk / N, from the window totals collected in pass 1.
        return cls(family, data.source_names, data.window_names, data.covariate_names,
                   window_sum / n_samples, source_sd, cov_mean, cov_sd, y_mean, y_sd)

    # Converts raw S_tilde_iwk to S_iwk using training mu_wk and sigma_k.
    def transform_windows(self, values):
        result = _float32(values, "window_scores")

        if result.ndim != 3 or result.shape[1:] != self.window_mean.shape:
            raise ValueError("Window-score block does not match the fitted layout.")

        # S_iwk = (S_tilde_iwk - mu_wk) / sigma_k.
        # The shared source SD preserves sum_w S_iwk = S_ik, up to rounding.
        result -= self.window_mean
        result /= self.source_sd

        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized window scores.")

        return result

    # Standardizes covariates using their training means and SDs.
    def transform_covariates(self, values):
        result = _float32(values, "covariates")

        if result.ndim != 2 or result.shape[1] != len(self.covariate_names):
            raise ValueError("Covariate block does not match the fitted layout.")

        result -= self.cov_mean
        result /= self.cov_sd

        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized covariates.")

        return result

    # Standardizes y with training values; binary phenotypes remain unchanged.
    def transform_outcome(self, values):
        result = _float32(values, "y")

        assert result.ndim == 1

        if self.family == "binomial" and not np.isin(values, (0, 1)).all():
            raise ValueError("Binomial outcomes must be exactly 0 or 1.")

        result -= np.float32(self.y_mean)
        result /= np.float32(self.y_sd)

        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized outcomes.")

        return result


class Design:
    """Blockwise construction of the BLocPRS design matrix.

    Each row contains an intercept, covariates C_ic, source scores S_ik
    and window contrasts Z_iwk: [1, C, S, Z]. Training means and SDs
    stay fixed during validation and prediction. Selected participants retain
    their supplied order. Standardization uses float32; model blocks use float64.
    """

    # Checks source/window/covariate order and selects participants without copying scores.
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

            assert step == 1

            self.rows = slice(start, stop, 1)
            self.n_samples = max(0, stop - start)
        else:
            rows = np.asarray(rows)

            if rows.dtype.kind == "b" and rows.shape == (data.n_samples,):
                rows = np.flatnonzero(rows)

            if (rows.ndim != 1 or not pd.api.types.is_integer_dtype(rows.dtype) or np.any(rows < 0)
                or np.any(rows >= data.n_samples) or len(np.unique(rows)) != len(rows)):
                raise ValueError("rows must be unique, in-range integer indices or an N-vector mask.")

            self.rows = rows.astype(np.int64, copy=True)
            self.n_samples = len(rows)

        assert self.n_samples >= 1

        self.n_covariates = data.n_covariates
        self.n_features = 1 + data.n_covariates + data.n_sources * (1 + data.n_windows)

    # Returns each block's output positions and corresponding input participant rows.
    def _blocks(self):
        for start in range(0, self.n_samples, self.block_size):
            stop = min(start + self.block_size, self.n_samples)
            if isinstance(self.rows, slice):
                rows = slice(self.rows.start + start, self.rows.start + stop)
            else:
                rows = self.rows[start:stop]

            yield start, stop, rows

    # Builds float64 (X, y) blocks in the selected participant order.
    # Returns y=None without reading phenotypes when y is absent or include_outcome=False.
    def iter_blocks(self, *, include_outcome=True):
        data = self.data
        first_genetic = 1 + self.n_covariates
        first_window = first_genetic + data.n_sources

        for start, stop, rows in self._blocks():

            # Builds X = [1, C, S, Z], with source columns S_ik and window columns Z_iwk.
            X = np.empty((stop - start, self.n_features), dtype=np.float64)
            X[:, 0] = 1
            X[:, 1:first_genetic] = self.scaler.transform_covariates(data.covariates[rows])

            # S_ik = sum_w S_iwk; Z_iwk = S_iwk - S_ik / W.
            # sum_w Z_iwk = 0 separates regional adjustments from genome-wide effects.
            windows = X[:, first_window:].reshape(-1, data.n_windows, data.n_sources)
            windows[:] = self.scaler.transform_windows(data.window_scores[rows])
            global_score = X[:, first_genetic:first_window]
            global_score[:] = windows.sum(axis=1, dtype=np.float64)
            windows -= global_score[:, None, :] * (1.0 / data.n_windows)

            if not include_outcome or data.y is None:
                y = None
            else:
                y = self.scaler.transform_outcome(data.y[rows]).astype(np.float64)

            if not np.isfinite(X).all():
                raise ValueError("Nonfinite model design.")

            yield X, y

    # Collects transformed phenotypes without reading PRS; returns None if absent.
    def outcome(self):
        if self.data.y is None:
            return None

        result = np.empty(self.n_samples, dtype=np.float64)

        for start, stop, rows in self._blocks():
            result[start:stop] = self.scaler.transform_outcome(self.data.y[rows])

        return result

    # Collects selected transformed N x C covariates without reading PRS.
    def covariates(self):
        result = np.empty((self.n_samples, self.n_covariates), dtype=np.float64)

        for start, stop, rows in self._blocks():
            result[start:stop] = self.scaler.transform_covariates(self.data.covariates[rows])

        return result
