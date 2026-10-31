"""Temporary, memory-mapped inputs for repeated model calculations.

Raw CV rows are staged once. Transformed designs retain float64 values and
are built separately for each training-fold scaler, then removed after use.
"""

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import numpy as np

from .data import PRSData, _ScoreView


logger = logging.getLogger(__name__)


# Check available temporary space before starting a collection of caches.
@contextmanager
def workspace(cache_dir=None, required_bytes=0):
    root = Path(cache_dir or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    required_bytes = int(required_bytes) + 64 * 1024**2

    if shutil.disk_usage(root).free < required_bytes:
        raise OSError(f"Model caches need approximately {required_bytes / 1024**3:.1f} GiB free in {root}. "
                      "Choose another cache_dir or --cache-dir.")

    with tempfile.TemporaryDirectory(prefix="blocprs_", dir=root) as directory:
        yield Path(directory)


# Copy only the sampled raw CV scores, preserving their values and row order.
@contextmanager
def stage_cv_data(data, rows, directory, block_size):
    started = perf_counter()
    path = Path(directory) / "cv_scores.npy"
    shape = (len(rows), data.n_windows, data.n_sources)
    scores = np.lib.format.open_memmap(path, mode="w+", dtype=data.window_scores.dtype, shape=shape)
    logger.info("CV input cache: %s participants", len(rows))

    try:
        # Resolve physical rows before scheduling reads; a source view may reorder both axes.
        source, source_rows, columns = data.window_scores, np.asarray(rows), None

        if isinstance(source, _ScoreView):
            source_rows, columns = source.rows[rows], source.sources
            source = source.scores

        read_order = np.argsort(source_rows, kind="stable")
        sorted_rows = source_rows[read_order]
        first = int(sorted_rows[0]) // block_size * block_size
        last = int(sorted_rows[-1]) + 1
        completed, next_report = 0, 25

        # Stage-5 score arrays are row-major. Copy contiguous blocks before selecting rows
        # so shared-storage reads stay sequential rather than following random CV indices.
        for start in range(first, last, block_size):
            stop = min(start + block_size, source.shape[0])
            left, right = np.searchsorted(sorted_rows, (start, stop))

            if left == right:
                continue

            block = np.array(source[start:stop], copy=True, order="C")
            selected = block[sorted_rows[left:right] - start]

            if columns is not None:
                selected = selected[:, :, columns]

            # Scatter locally into the original sampled order; folds and scaling are unchanged.
            scores[read_order[left:right]] = selected
            completed += right - left
            del block, selected
            percent = 100 * completed // len(rows)

            if percent >= next_report or completed == len(rows):
                logger.info("CV input cache: %s/%s participants staged", completed, len(rows))
                next_report = (percent // 25 + 1) * 25

        scores.flush()
        scores._mmap.close()
        scores = np.load(path, mmap_mode="r", allow_pickle=False)
        staged = PRSData(scores, data.source_names, data.window_names, y=data.y[rows],
                         covariates=data.covariates[rows], covariate_names=data.covariate_names,
                         sample_ids=None if data.sample_ids is None else data.sample_ids[rows])
        logger.info("CV input cache ready in %.1fs", perf_counter() - started)
        yield staged
    finally:
        scores._mmap.close()
        path.unlink(missing_ok=True)


class CachedDesign:
    """Read an already transformed design without repeating standardization."""

    # Keep the same layout and participant blocks as the original design.
    def __init__(self, X, y, design):
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape
        self.n_covariates = design.n_covariates
        self.block_size = design.block_size

    # Read a contiguous block; concurrent workers never modify these arrays.
    def read_block(self, start, stop):
        return self.X[start:stop], self.y[start:stop]

    # Iterate in the original row order to retain the same reduction order.
    def iter_blocks(self, *, include_outcome=True):
        for start in range(0, self.n_samples, self.block_size):
            stop = min(start + self.block_size, self.n_samples)
            yield self.X[start:stop], self.y[start:stop] if include_outcome else None

    # Outcomes are already transformed using the associated training scaler.
    def outcome(self):
        return self.y

    # Exclude the intercept and genetic columns from evaluation covariates.
    def covariates(self):
        return self.X[:, 1:1 + self.n_covariates]


# Cache a transformed design once; retain float64 arithmetic and remove it on exit.
@contextmanager
def cached_design(design, directory, name):
    started = perf_counter()
    folder = Path(directory) / name
    folder.mkdir()
    X = y = None
    logger.info("%s design cache: N=%s, features=%s", name, design.n_samples, design.n_features)

    try:
        X = np.lib.format.open_memmap(folder / "X.npy", mode="w+", dtype=np.float64,
                                      shape=(design.n_samples, design.n_features))
        y = np.lib.format.open_memmap(folder / "y.npy", mode="w+", dtype=np.float64,
                                      shape=(design.n_samples,))
        offset = 0
        for values, outcome in design.iter_blocks():
            stop = offset + len(values)

            if (outcome is None or stop > design.n_samples or values.shape != (len(values), design.n_features)
                or outcome.shape != (len(values),)):
                raise ValueError("A training/validation cache needs aligned outcomes and design rows.")

            X[offset:stop], y[offset:stop] = values, outcome
            offset = stop

        if offset != design.n_samples:
            raise ValueError("The design cache has incomplete participant rows.")

        # Do not retain the last full design block while the cached fit runs.
        del values, outcome
        X.flush()
        y.flush()
        X._mmap.close()
        y._mmap.close()
        X = np.load(folder / "X.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(folder / "y.npy", mmap_mode="r", allow_pickle=False)
        logger.info("%s design cache ready in %.1fs", name, perf_counter() - started)
        yield CachedDesign(X, y, design)
    finally:
        for array in (X, y):
            if array is not None:
                array._mmap.close()
        shutil.rmtree(folder)
