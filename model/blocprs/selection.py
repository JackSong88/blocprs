"""Participant selection and reproducible training subsets."""

from pathlib import Path

import numpy as np
import pandas as pd


# Read one- or two-column ID files, with an optional header.
def read_ids(path):
    table = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, keep_default_na=False)

    if table.shape[1] not in (1, 2):
        raise ValueError("ID files need one IID column or two FID/IID columns.")

    header = tuple(value.lstrip("#").upper() for value in table.iloc[0])
    if header in (("IID",), ("ID",), ("FID", "IID")):
        table = table.iloc[1:]

    if table.empty or table.isna().any().any() or table.eq("").any().any() or table.duplicated().any():
        raise ValueError(f"Empty, missing or duplicate participant IDs: {path}")

    table.columns = ["IID"] if table.shape[1] == 1 else ["FID", "IID"]
    return table.reset_index(drop=True)


# Match IDs within the supplied population, retaining the ID-file order.
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


# Read participant metadata and the small covariate array, without reading PRSs.
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

        # Do not silently replace a saved metadata field with a covariate.
        if name in table and not np.array_equal(table[name].to_numpy(), covariates[:, index]):
            raise ValueError(f"Saved participant metadata and covariates disagree for {name}.")

        table[name] = covariates[:, index]

    return table if split is None else table.loc[table["split"].eq(split)].copy()


# Filter recorded values; non-EUR means every recorded ancestry except EUR.
def filter_participants(table, ancestry=None, sex=None, filters=None):
    keep = np.ones(len(table), dtype=bool)
    if ancestry:
        values = [ancestry] if isinstance(ancestry, str) else ancestry
        ancestry_keep = table["Ancestry"].isin(values).to_numpy()
        if "non-EUR" in values:
            ancestry_keep |= table["Ancestry"].notna().to_numpy() & table["Ancestry"].ne("EUR").to_numpy()
        keep &= ancestry_keep

    selections = [] if filters is None else list(filters)
    if sex is not None:
        selections.append(("Sex", [sex] if isinstance(sex, (str, int, float)) else sex))

    for selection in selections:
        name, values = selection.split("=", 1) if isinstance(selection, str) else selection
        values = [values] if isinstance(values, str) else list(values)
        column = table[name]
        if pd.api.types.is_numeric_dtype(column):
            values = [float(value) for value in values]
        keep &= column.isin(values).to_numpy()

    result = table.loc[keep].copy()

    if result.empty:
        raise ValueError("No participants remain after filtering.")

    return result


# Form joint strata, including binary phenotype classes unless disabled.
def stratum_labels(table, columns=None, binary_default=True):
    columns = [] if columns is None else list(columns)
    binary = set(table["phenotype"].dropna().unique()) == {0, 1}
    if binary_default and binary and "phenotype" not in columns:
        columns.insert(0, "phenotype")
    if not columns:
        return None

    values = table[columns]

    if values.isna().any().any():
        raise ValueError("Stratification columns contain missing values.")

    return pd.factorize(pd.MultiIndex.from_frame(values), sort=True)[0]


# Draw exact-size subsets; nested subsets are prefixes of one stratified order.
def sample_subsets(n_samples, sizes, labels=None, nested=False, seed=42):
    sizes = np.asarray(sizes)

    if (sizes.ndim != 1 or sizes.dtype.kind not in "iu" or not len(sizes)
        or len(sizes) != len(set(sizes)) or np.any(sizes < 1) or np.any(sizes > n_samples)):
        raise ValueError("Sample sizes must be distinct positive integers within the training population.")

    if labels is not None:
        labels = np.asarray(labels)

        if labels.shape != (n_samples,):
            raise ValueError("Strata must describe the full sampling population.")

    sizes = sizes.tolist()
    rng = np.random.default_rng(seed)
    groups = ([np.arange(n_samples)] if labels is None else
              [np.flatnonzero(labels == label) for label in np.unique(labels)])
    counts = np.array([len(group) for group in groups])
    if nested:

        # Interleave shuffled strata in proportion to their population sizes.
        shuffled = [rng.permutation(group) for group in groups]

        # Midpoint ranks put each stratum on the same 0-to-1 scale.
        # Taking prefixes of the combined order makes all requested subsets nested.
        priorities = np.concatenate([(np.arange(count) + 0.5) / count for count in counts])
        order = np.concatenate(shuffled)[np.argsort(priorities, kind="stable")]
        return {size: order[:size].copy() for size in sizes}

    subsets = {}
    for size in sizes:

        # Largest remainders preserve proportions while giving the exact total.
        expected = counts * (size / n_samples)
        allocated = np.floor(expected).astype(int)
        remainder = size - int(allocated.sum())
        allocated[np.argsort(-(expected - allocated), kind="stable")[:remainder]] += 1
        selected = np.concatenate([rng.choice(group, count, replace=False)
                                   for group, count in zip(groups, allocated)])
        subsets[size] = rng.permutation(selected)
    return subsets
