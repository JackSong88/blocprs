"""Read numerical PRS arrays and their labels, or existing score tables."""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


READ_BUFFER_BYTES = 32 * 1024**2


class ScoreTable:
    """Raw scores with explicit participant, window and source order."""

    # Read labels without loading the numerical matrix.
    def __init__(self, path):
        self.path = Path(path)
        self.array = None
        self.ids = None

        if self.path.is_dir():
            self.ids = pd.read_csv(self.path / "participants.tsv", sep="\t", dtype=str,
                                   keep_default_na=False)[["FID", "IID"]]
            features = pd.read_csv(self.path / "features.tsv", sep="\t", dtype=str,
                                    keep_default_na=False)
            windows = features["window"].drop_duplicates().tolist()
            sources = features["source"].drop_duplicates().tolist()
            pairs = [(window, source) for window in windows for source in sources]

            if list(zip(features["window"], features["source"])) != pairs:
                raise ValueError("Score features must describe a complete window-major grid.")

            self.columns = features["column"].tolist()
            expected = [source if window == "genomewide" else f"{source}_{window}"
                        for window, source in pairs]

            if self.columns != expected:
                raise ValueError("Score column labels disagree with their windows or sources.")

            self.array = np.load(self.path / "scores.npy", mmap_mode="r", allow_pickle=False)

            if (self.array.shape != (len(self.ids), len(windows), len(sources))
                    or self.array.dtype not in (np.dtype("float32"), np.dtype("float64"))):
                raise ValueError("Score array dimensions or precision disagree with its labels.")

            if self.ids.duplicated().any() or self.ids.apply(lambda column: column.str.strip().eq("").any()).any():
                raise ValueError("Score participant IDs must be nonempty and unique.")

        else:
            opener = gzip.open if self.path.suffix == ".gz" else open
            with opener(self.path, "rt") as handle:
                header = handle.readline().rstrip("\r\n").split("\t")

            if len(header) != len(set(header)) or any(not name or name != name.strip() for name in header):
                raise ValueError("Score headers must be unique, nonempty and without surrounding whitespace.")

            self.columns = [name for name in header if name not in ("FID", "IID")]

        if not self.columns or len(self.columns) != len(set(self.columns)):
            raise ValueError("Score columns must be nonempty and unique.")

    # Binary scores keep IDs separately; existing TSV inputs remain supported.
    def iter_ids(self, chunk_size=100000):
        if self.ids is not None:
            for start in range(0, len(self.ids), chunk_size):
                yield self.ids.iloc[start:start + chunk_size]
        else:
            with pd.read_csv(self.path, sep="\t", usecols=["FID", "IID"], dtype=str,
                             chunksize=chunk_size) as chunks:
                yield from chunks

    # Read only a bounded set of rows, preserving the requested feature order.
    def iter_chunks(self, columns=None, chunk_size=65536, dtype=np.float64):
        columns = self.columns if columns is None else list(columns)
        positions = {name: index for index, name in enumerate(self.columns)}
        selected = np.array([positions[name] for name in columns], dtype=int)
        chunk_size = max(1, min(chunk_size, READ_BUFFER_BYTES // (np.dtype(dtype).itemsize * len(columns))))

        if self.array is None:
            types = {"FID": str, "IID": str, **{name: dtype for name in columns}}
            with pd.read_csv(self.path, sep="\t", usecols=["FID", "IID", *columns],
                             dtype=types, chunksize=chunk_size) as chunks:
                for frame in chunks:
                    yield frame[["FID", "IID"]], frame[columns].to_numpy(dtype=dtype)
            return

        n_sources = self.array.shape[2]
        window_indices, source_indices = selected // n_sources, selected % n_sources
        for start in range(0, len(self.ids), chunk_size):
            stop = min(start + chunk_size, len(self.ids))
            values = np.asarray(self.array[start:stop, window_indices, source_indices], dtype=dtype)
            yield self.ids.iloc[start:stop], values
