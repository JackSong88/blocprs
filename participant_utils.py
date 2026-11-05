"""Shared participant IDs, metadata and filtering."""

from pathlib import Path

import numpy as np
import pandas as pd


# Reads one- or two-column ID files, with an optional header.
def read_ids(path):
    table = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, keep_default_na=False)

    if table.shape[1] not in (1, 2):
        raise ValueError("ID files need one IID column or two FID/IID columns.")

    # check if header
    header = tuple(value.lstrip("#").upper() for value in table.iloc[0])

    if header in (("IID",), ("ID",), ("FID", "IID")):
        table = table.iloc[1:]

    if table.empty or table.isna().any().any() or table.eq("").any().any() or table.duplicated().any():
        raise ValueError(f"Empty, missing or duplicate participant IDs: {path}")

    table.columns = ["IID"] if table.shape[1] == 1 else ["FID", "IID"]

    return table.reset_index(drop=True)


# Matches IDs within the supplied population, retaining the ID-file order.
def id_positions(participants, ids_file):
    ids = read_ids(ids_file)
    columns = ids.columns.tolist()
    available = pd.MultiIndex.from_frame(participants[columns])

    if available.has_duplicates:
        raise ValueError("Participant IDs are ambiguous; use unique FID/IID pairs.")

    rows = available.get_indexer(pd.MultiIndex.from_frame(ids))

    # pandas uses -1 for absent IDs; using those as array indices would select the last row.
    if np.any(rows < 0):
        raise ValueError(f"{int((rows < 0).sum())} requested IDs are absent from the selected split/population.")

    return rows


# Reads participant metadata and the small covariate array, without reading PRSs.
def read_participants(path, split=None):
    path = Path(path)
    table = pd.read_csv(path / "participants.tsv", sep="\t", dtype={"FID": str, "IID": str},
                        float_precision="round_trip")
    names = pd.read_csv(path / "covariates.tsv", sep="\t")["covariate"].tolist()
    covariates = np.load(path / "covariates.npy", mmap_mode="r", allow_pickle=False)

    ids = table[["FID", "IID"]]

    if ids.isna().any().any() or any(ids[name].str.strip().eq("").any() for name in ids) or ids.duplicated().any():
        raise ValueError("Participants contain missing or duplicate IDs.")

    if covariates.shape != (len(table), len(names)):
        raise ValueError("Participant and covariate rows are inconsistent.")

    for index, name in enumerate(names):
        table[name] = covariates[:, index]

    return table if split is None else table.loc[table["split"].eq(split)].copy()


# Filters recorded values; non-EUR means every recorded ancestry except EUR.
def filter_participants(table, ancestry=None, sex=None, filters=None):
    # Selects the requested ancestries.
    if ancestry:
        if isinstance(ancestry, str):
            ancestry = [ancestry]

        matches = table["Ancestry"].isin(ancestry)

        # Includes all recorded ancestries other than EUR.
        if "non-EUR" in ancestry:
            non_eur = table["Ancestry"].notna() & (table["Ancestry"] != "EUR")
            matches = matches | non_eur

        table = table.loc[matches]

    # Selects the requested sex values using the recorded coding.
    if sex is not None:
        if isinstance(sex, (str, int, float)):
            sex = [sex]

        if pd.api.types.is_numeric_dtype(table["Sex"]):
            sex = [float(value) for value in sex]

        table = table.loc[table["Sex"].isin(sex)]

    # Applies any additional column/value filters to the remaining participants.
    selections = [] if filters is None else list(filters)

    for selection in selections:
        if isinstance(selection, str):
            name, values = selection.split("=", 1)
        else:
            name, values = selection

        if isinstance(values, str):
            values = [values]
        else:
            values = list(values)

        column = table[name]

        if pd.api.types.is_numeric_dtype(column):
            values = [float(value) for value in values]

        table = table.loc[column.isin(values)]

    assert not table.empty, "No participants remain after filtering."

    return table.copy()
