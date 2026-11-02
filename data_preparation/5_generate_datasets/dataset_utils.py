# Adapts MoEPRS for BLocPRS: https://github.com/shz9/moe-prs-paper

"""Reads saved PRSs and clinical metadata and matches participant IDs."""

from pathlib import Path

import numpy as np
import pandas as pd


READ_BUFFER_BYTES = 32 * 1024**2


class ScoreTable:
    """Reads a scores.npy array with participants.tsv and features.tsv labels.

    Uses shape (participants, windows, PRSs). Reads labels immediately and
    numerical scores from disk as needed.
    """

    # Reads labels without loading the numerical matrix.
    def __init__(self, path):
        self.path = Path(path)

        self.ids = pd.read_csv(self.path / "participants.tsv", sep="\t", dtype=str,
                               keep_default_na=False)[["FID", "IID"]]
        self.features = pd.read_csv(self.path / "features.tsv", sep="\t", dtype=str,
                                    keep_default_na=False)
        self.windows = self.features["window"].drop_duplicates().tolist()
        self.sources = self.features["source"].drop_duplicates().tolist()
        self.columns = self.features["column"].tolist()

        # Checks that labels cover every window/PRS pair in array order.
        pairs = [(window, source) for window in self.windows for source in self.sources]
        expected_columns = [source if window == "genomewide" else f"{source}_{window}"
                            for window, source in pairs]

        if (list(zip(self.features["window"], self.features["source"])) != pairs
                or self.columns != expected_columns or len(set(self.columns)) != len(self.columns)
                or not self.columns):
            raise ValueError("Score labels must describe every window/PRS pair exactly once, in array order.")

        if self.ids.duplicated().any() or any(self.ids[column].str.strip().eq("").any() for column in self.ids):
            raise ValueError("Score participant IDs must be nonempty and unique.")

        self.array = np.load(self.path / "scores.npy", mmap_mode="r", allow_pickle=False)

        if (self.array.shape != (len(self.ids), len(self.windows), len(self.sources))
                or self.array.dtype not in (np.dtype("float32"), np.dtype("float64"))):
            raise ValueError("Score array dimensions or precision disagree with its labels.")

    # Reads bounded row blocks in the requested feature order.
    def iter_chunks(self, columns=None, chunk_size=65536, dtype=np.float64):
        columns = self.columns if columns is None else list(columns)
        positions = {name: index for index, name in enumerate(self.columns)}
        selected = np.array([positions[name] for name in columns], dtype=int)
        chunk_size = max(1, min(chunk_size, READ_BUFFER_BYTES // (np.dtype(dtype).itemsize * len(columns))))

        n_sources = self.array.shape[2]

        # Converts window-major column positions into window and PRS indices.
        window_indices, source_indices = selected // n_sources, selected % n_sources

        for start in range(0, len(self.ids), chunk_size):
            stop = min(start + chunk_size, len(self.ids))
            values = np.asarray(self.array[start:stop, window_indices, source_indices], dtype=dtype)
            yield self.ids.iloc[start:stop], values

    # Reads a few feature columns across all participants.
    def iter_feature_chunks(self, dtype=np.float32):
        n, windows, sources = self.array.shape
        width = max(1, READ_BUFFER_BYTES // (n * max(self.array.dtype.itemsize, np.dtype(dtype).itemsize)))

        # Preserves extraction's Fortran order to read feature columns sequentially.
        matrix = self.array.reshape(n, windows * sources, order="F")
        column_order = np.arange(windows * sources).reshape(windows, sources).ravel(order="F")

        for start in range(0, len(column_order), width):
            stop = min(start + width, len(column_order))

            # Copies a contiguous disk interval once for row selection in memory.
            values = np.array(matrix[:, start:stop], dtype=dtype, order="F", copy=True)
            yield [self.columns[index] for index in column_order[start:stop]], values


def participant_index(table, label):
    ids = table[["FID", "IID"]]

    # Checks for missing participant IDs.
    if ids.isna().any().any() or any(ids[column].str.strip().eq("").any() for column in ids):
        raise ValueError(f"{label} contains missing participant IDs.")

    index = pd.MultiIndex.from_frame(ids)

    # Checks for duplicate FID/IID pairs.
    if index.has_duplicates:
        raise ValueError(f"{label} contains duplicate FID/IID pairs.")

    return index


def read_clinical_table(path, columns):
    table = pd.read_csv(path, sep=r"\s+", header=None, dtype={0: str, 1: str}, skip_blank_lines=False)
    table.columns = columns
    table.index = participant_index(table, str(path))

    for column in columns[2:]:
        table[column] = pd.to_numeric(table[column], errors="raise")

    # Rejects infinity; missing clinical records are excluded later.
    if np.isinf(table[columns[2:]].to_numpy(dtype=float)).any():
        raise ValueError(f"{path} contains infinite clinical values.")

    return table.drop(columns=["FID", "IID"])


# Uses complete phenotype/covariate rows and labels missing ancestry as OTH.
def read_metadata(phenotype_file, covariates_file, ancestry_file, num_pcs, family):
    phenotype = read_clinical_table(phenotype_file, ["FID", "IID", "phenotype"])
    covariate_names = ["Sex"] + [f"PC{i}" for i in range(1, num_pcs + 1)] + ["Age"]
    covariates = read_clinical_table(covariates_file, ["FID", "IID"] + covariate_names)

    ancestry = pd.read_csv(ancestry_file, sep="\t", dtype={"FID": str, "IID": str, "Ancestry": str})
    ancestry.index = participant_index(ancestry, str(ancestry_file))

    # Keeps complete phenotype and covariate records for the same IDs.
    available = phenotype.dropna().join(covariates.dropna(), how="inner")

    if family == "binomial" and not available["phenotype"].isin([0, 1]).all():
        raise ValueError("Binomial phenotypes must be coded exactly 0 or 1.")

    available = available.join(ancestry[["Ancestry"]], how="left")
    available["Ancestry"] = available["Ancestry"].fillna("OTH")

    return available, sorted(covariate_names)


# Orders windows numerically and PRS names alphabetically, using the saved labels.
def feature_layout(table):
    features = table.features.copy()

    if table.windows == ["genomewide"]:
        return features.sort_values("source").reset_index(drop=True)

    features["window_number"] = features["window"].str.removeprefix("w").astype(int)

    if features.duplicated(["window_number", "source"]).any():
        raise ValueError("Window labels refer to repeated window/PRS pairs.")

    features = features.sort_values(["window_number", "source"])
    features["window"] = features["window_number"].map(lambda number: f"w{number:04d}")

    return features.drop(columns="window_number").reset_index(drop=True)


# Keeps participants with scores and complete clinical records, in score-file order.
def match_score_participants(table, metadata):
    ids = pd.MultiIndex.from_frame(table.ids)
    selected = ids[ids.isin(metadata.index)]

    if not len(selected):
        raise ValueError("No participants have scores, phenotype and complete covariates in common.")

    return metadata.loc[selected].copy(), len(ids), len(metadata) - len(selected)
