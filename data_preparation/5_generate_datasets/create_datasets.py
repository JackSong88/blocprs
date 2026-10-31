# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Join raw PRS and cohort inputs using saved splits, then save a memory-mapped dataset."""

import argparse
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from blocprs.selection import read_ids, stratum_labels

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_io import ScoreTable

logger = logging.getLogger(__name__)


def participant_index(table, label):
    ids = table[["FID", "IID"]]

    # Missing IDs check.
    if ids.isna().any().any() or any(ids[column].str.strip().eq("").any() for column in ids):
        raise ValueError(f"{label} contains missing participant IDs.")

    index = pd.MultiIndex.from_frame(ids)

    # Duplicate IDs check.
    if index.has_duplicates:
        raise ValueError(f"{label} contains duplicate FID/IID pairs.")

    return index


def read_clinical_table(path, columns):
    table = pd.read_csv(path, sep=r"\s+", header=None, dtype={0: str, 1: str}, skip_blank_lines=False)
    table.columns = columns
    table.index = participant_index(table, str(path))
    for column in columns[2:]:
        table[column] = pd.to_numeric(table[column], errors="raise")

        if np.isinf(table[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{path} contains infinite values in {column}.")

    return table.drop(columns=["FID", "IID"])


# Use complete phenotype/covariate rows; missing ancestry retains the original OTH label.
def read_metadata(phenotype_file, covariates_file, ancestry_file, num_pcs, family):
    phenotype = read_clinical_table(phenotype_file, ["FID", "IID", "phenotype"])
    covariate_names = ["Sex"] + [f"PC{i}" for i in range(1, num_pcs + 1)] + ["Age"]
    covariates = read_clinical_table(covariates_file, ["FID", "IID"] + covariate_names)
    ancestry = pd.read_csv(ancestry_file, sep="\t", dtype={"FID": str, "IID": str, "Ancestry": str})
    ancestry.index = participant_index(ancestry, str(ancestry_file))
    available = phenotype.dropna().join(covariates.dropna(), how="inner")

    if family == "binomial" and not available["phenotype"].isin([0, 1]).all():
        raise ValueError("Binomial phenotypes must be coded exactly 0 or 1.")

    available = available.join(ancestry[["Ancestry"]], how="left")
    available["Ancestry"] = available["Ancestry"].fillna("OTH")
    return available, sorted(covariate_names)


# Map named score columns onto a complete window-by-source grid without adding zeros.
def feature_layout(scores_file):
    columns = ScoreTable(scores_file).columns

    if not columns:
        raise ValueError("No PRS columns were found.")

    parsed = [re.fullmatch(r"(.+?)_{1,2}w(\d+)", column) for column in columns]

    if any(parsed) and not all(parsed):
        raise ValueError("Do not mix genome-wide and windowed score columns in one dataset.")

    if all(parsed):
        pairs = [(int(match.group(2)), match.group(1)) for match in parsed]
        windows = sorted({window for window, _ in pairs})
        sources = sorted({source for _, source in pairs})
        mapping = dict(zip(pairs, columns))

        if len(mapping) != len(columns) or len(mapping) != len(windows) * len(sources):
            raise ValueError("Window/source columns are not a complete unique grid. Re-extract with the canonical window and scorefile schema; missing columns are not padded here.")

        rows = [(f"w{window:04d}", source, mapping[window, source]) for window in windows for source in sources]
    else:
        sources = sorted(columns)
        rows = [("genomewide", source, source) for source in sources]
    return pd.DataFrame(rows, columns=["window", "source", "column"])


# Read only score IDs first, so large score arrays can be allocated to their final size.
def score_participants(scores_file, metadata):
    seen = set()
    selected = []
    for chunk in ScoreTable(scores_file).iter_ids(chunk_size=100000):
        ids = participant_index(chunk, "Scores")
        keys = set(ids.tolist())

        if not seen.isdisjoint(keys):
            raise ValueError("Scores contain duplicate participant IDs across row blocks.")

        seen.update(keys)
        selected.extend(ids[ids.isin(metadata.index)].tolist())

    if not selected:
        raise ValueError("No participants have scores, phenotype and complete covariates in common.")

    index = pd.MultiIndex.from_tuples(selected, names=["FID", "IID"])
    return metadata.loc[index].copy(), len(seen), len(metadata) - len(selected)


def read_split_ids(path):
    table = read_ids(path)
    keys = table.iloc[:, 0].tolist() if table.shape[1] == 1 else list(table.itertuples(index=False, name=None))
    return keys, table.shape[1]


# Preserve saved split order; otherwise split 70/30, stratifying binary traits.
def assign_splits(metadata, family, train_ids_file=None, test_ids_file=None, all_participants=False,
                  test_fraction=0.3, seed=7209, stratify=None):

    if (train_ids_file is None) != (test_ids_file is None):
        raise ValueError("Supply both train and test ID files together.")

    if all_participants and train_ids_file is not None:
        raise ValueError("Choose saved split files or all participants, not both.")

    if stratify and (all_participants or train_ids_file is not None):
        raise ValueError("--stratify applies only when creating a new random train/test split.")

    missing = 0
    if all_participants:
        result = metadata.copy()
        result["split"] = "all"
        return result, missing
    if train_ids_file is not None:
        train_ids, train_width = read_split_ids(train_ids_file)
        test_ids, test_width = read_split_ids(test_ids_file)

        if train_width != test_width or not set(train_ids).isdisjoint(test_ids):
            raise ValueError("Train/test files must use the same ID format and have no overlap.")

        index = metadata.index if train_width == 2 else pd.Index(metadata.index.get_level_values("IID"))

        if index.has_duplicates:
            raise ValueError("IIDs are not unique; supply two-column FID/IID split files.")

        if not set(index).issubset(set(train_ids).union(test_ids)):
            raise ValueError("Some eligible participants are absent from both split files.")

        # get_indexer returns -1 for absent IDs; count these before selecting any rows.
        train_rows, test_rows = index.get_indexer(train_ids), index.get_indexer(test_ids)
        missing = int((train_rows < 0).sum() + (test_rows < 0).sum())

        if missing:
            raise ValueError(f"{missing} saved split IDs are absent from the eligible dataset; check inputs and eligibility.")
    else:

        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1).")

        labels = stratum_labels(metadata, stratify, binary_default=family == "binomial")
        train_rows, test_rows = train_test_split(np.arange(len(metadata)), test_size=test_fraction,
                                                stratify=labels, random_state=seed)

    if not len(train_rows) or not len(test_rows):
        raise ValueError("Training and test sets must both contain participants.")

    if family == "binomial" and set(metadata.iloc[train_rows]["phenotype"]) != {0, 1}:
        raise ValueError("Binary training data must contain both cases and controls.")

    result = metadata.iloc[np.concatenate([train_rows, test_rows])].copy()
    result["split"] = ["train"] * len(train_rows) + ["test"] * len(test_rows)
    return result, missing


# Transpose column-major extraction output locally, then publish it with sequential I/O.
def write_feature_ordered_scores(table, metadata, features, output_dir):
    columns = features["column"].tolist()
    column_positions = {name: index for index, name in enumerate(columns)}
    rows = participant_index(table.ids, "Scores").get_indexer(metadata.index)

    if (rows < 0).any():
        raise ValueError("Selected participants are absent from the input scores.")

    shape = (len(metadata), features["window"].nunique(), features["source"].nunique())
    started = last_log = monotonic()
    written = np.zeros(len(columns), dtype=bool)

    # Temporary storage must have room for one uncompressed participant-major matrix.
    with TemporaryDirectory(prefix="blocprs_dataset_", dir=os.environ.get("SLURM_TMPDIR")) as work:
        required = int(np.prod(shape)) * np.dtype(np.float32).itemsize + 1024**2

        if shutil.disk_usage(work).free < required:
            raise OSError(f"Temporary storage needs at least {required / 1024**3:.1f} GiB for this dataset.")

        path = Path(work) / "scores.npy"
        logger.info("Assembling participant-major scores in temporary storage: N=%s, features=%s",
                    len(metadata), len(columns))
        scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        matrix = scores.reshape(len(metadata), -1)
        try:
            for names, values in table.iter_feature_chunks():

                if not np.isfinite(values).all():
                    raise ValueError("Score entries must be finite; missing values are not filled with zero.")

                positions = np.array([column_positions[name] for name in names])
                matrix[:, positions] = values[rows]
                written[positions] = True

                if monotonic() - last_log >= 60:
                    logger.info("Assembling scores: %s/%s features (%.1f%%); elapsed=%.1fs",
                                written.sum(), len(columns), 100 * written.mean(), monotonic() - started)
                    last_log = monotonic()

            if not written.all():
                raise ValueError("Input scores do not cover every requested feature.")

            scores.flush()
        finally:
            del matrix, scores

        logger.info("Copying completed scores to output: %.2f GiB", path.stat().st_size / 1024**3)
        shutil.copyfile(path, output_dir / "scores.npy")
    logger.info("Score writing complete: elapsed=%.1fs", monotonic() - started)


# Stream participant-major inputs in row blocks; transpose column-major inputs locally.
def write_arrays(scores_file, metadata, features, covariate_names, output_dir):
    columns = features["column"].tolist()
    n_windows, n_sources = features["window"].nunique(), features["source"].nunique()
    table = ScoreTable(scores_file)

    if table.array is not None and table.array.flags.f_contiguous and n_windows > 1:
        write_feature_ordered_scores(table, metadata, features, output_dir)
    else:
        write_row_ordered_scores(table, metadata, columns, n_windows, n_sources, output_dir)

    covariates = metadata[covariate_names].to_numpy(dtype=np.float32)

    if not np.isfinite(covariates).all():
        raise ValueError("Covariates must be finite and representable as float32.")

    np.save(output_dir / "covariates.npy", covariates, allow_pickle=False)
    metadata[["phenotype", "Ancestry", "split"]].reset_index().to_csv(output_dir / "participants.tsv", sep="\t", index=False)
    features.to_csv(output_dir / "features.tsv", sep="\t", index=False)
    pd.DataFrame({"covariate": covariate_names}).to_csv(output_dir / "covariates.tsv", sep="\t", index=False)


# Keep bounded row reads for participant-major arrays and existing text inputs.
def write_row_ordered_scores(table, metadata, columns, n_windows, n_sources, output_dir):
    scores = np.lib.format.open_memmap(output_dir / "scores.npy", mode="w+", dtype=np.float32,
                                     shape=(len(metadata), n_windows, n_sources))
    written = np.zeros(len(metadata), dtype=bool)
    chunk_size = max(1, min(10000, 32 * 1024**2 // (4 * len(columns))))
    chunks = table.iter_chunks(columns=columns, chunk_size=chunk_size, dtype=np.float32)
    started = last_log = monotonic()
    completed = 0
    logger.info("Writing scores: N=%s, windows=%s, sources=%s", len(metadata), n_windows, n_sources)
    try:
        for block, (chunk_ids, values) in enumerate(chunks, start=1):
            ids = participant_index(chunk_ids, "Scores")

            if not np.isfinite(values).all():
                raise ValueError("Score entries must be finite; missing or malformed values are not filled with zero.")

            # Map streamed rows into the saved train/test order, not score-file order.
            positions = metadata.index.get_indexer(ids)
            keep = positions >= 0
            selected = positions[keep]

            if written[selected].any():
                raise ValueError("Repeated score participants encountered during array writing.")

            scores[selected] = values[keep].reshape(-1, n_windows, n_sources)
            written[selected] = True
            completed += len(selected)

            if monotonic() - last_log >= 60:
                logger.info("Writing scores: %s/%s participants (%.1f%%); elapsed=%.1fs",
                            completed, len(metadata), 100 * completed / len(metadata), monotonic() - started)
                last_log = monotonic()

            # Flush completed blocks so dirty mapped pages do not accumulate.
            if block % 16 == 0:
                scores.flush()

        if not written.all():
            raise ValueError("The score file changed between ID selection and score writing.")

        scores.flush()
    finally:
        del scores

    logger.info("Score writing complete: elapsed=%.1fs", monotonic() - started)


def create_datasets(scores_file, phenotype_file, covariates_file, ancestry_file, output_dir, family,
                    num_pcs=10, train_ids_file=None, test_ids_file=None, all_participants=False):

    if family not in ("gaussian", "binomial") or num_pcs < 0:
        raise ValueError("Use gaussian/binomial family and a nonnegative number of PCs.")

    if not all_participants and (train_ids_file is None or test_ids_file is None):
        raise ValueError("Supply train/test ID files from create_splits.py, or use --all-participants for transfer data.")

    output_dir = Path(output_dir)

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing dataset: {output_dir}")

    started = monotonic()
    logger.info("Preparing dataset: scores=%s; family=%s", scores_file, family)
    features = feature_layout(scores_file)
    metadata, covariate_names = read_metadata(phenotype_file, covariates_file, ancestry_file, num_pcs, family)
    metadata, score_count, missing_scores = score_participants(scores_file, metadata)

    # Reuse the same participant membership and order for every score representation.
    metadata, missing_split_ids = assign_splits(metadata, family, train_ids_file, test_ids_file,
                                               all_participants)
    logger.info("Selected N=%s; splits=%s; windows=%s; sources=%s", len(metadata),
                metadata["split"].value_counts().to_dict(), features["window"].nunique(), features["source"].nunique())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=".dataset_") as temporary:
        temporary = Path(temporary)
        write_arrays(scores_file, metadata, features, covariate_names, temporary)

        # Publish only complete files into a newly created directory.
        output_dir.mkdir()
        published = []
        try:
            for path in temporary.iterdir():
                destination = output_dir / path.name
                os.link(path, destination)
                published.append(destination)
        except BaseException:
            for path in published:
                path.unlink()
            output_dir.rmdir()
            raise
    logger.info("Dataset saved: %s; elapsed=%.1fs", output_dir, monotonic() - started)
    return {"participants": len(metadata), "score_rows": score_count, "eligible_without_scores": missing_scores,
            "missing_split_ids": missing_split_ids, "splits": metadata["split"].value_counts().to_dict()}


# Load a saved dataset through the model package.
def load_dataset(path, split="train", ids_file=None):
    from blocprs import load_dataset as load

    return load(path, split=split, ids_file=ids_file)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-file", required=True, help="Binary raw-score directory, or tab-separated scores with FID/IID (optional gzip compression).")
    parser.add_argument("--phenotype-file", required=True, help="Headerless FID, IID, phenotype table.")
    parser.add_argument("--covariates-file", required=True, help="Headerless FID, IID, Sex, PC1..PCN, Age table.")
    parser.add_argument("--ancestry-file", required=True, help="TSV with FID, IID and Ancestry headers.")
    parser.add_argument("--output-dir", required=True, help="New dataset directory; existing directories are not replaced.")
    parser.add_argument("--family", required=True, choices=("gaussian", "binomial"))
    parser.add_argument("--num-pcs", type=int, default=10, help="Number of PCs (default: 10).")
    parser.add_argument("--train-ids-file", help="Saved IID list or FID/IID pairs, preserving file order.")
    parser.add_argument("--test-ids-file", help="Saved disjoint test IDs in the same format as the train file.")
    parser.add_argument("--all-participants", action="store_true", help="Save all eligible transfer participants without making a new split.")
    args = parser.parse_args()
    try:
        result = create_datasets(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {result['participants']} participants to {args.output_dir}; split counts: {result['splits']}.", flush=True)
    print(f"Score rows read: {result['score_rows']}; otherwise eligible participants without scores: {result['eligible_without_scores']}; absent saved split IDs: {result['missing_split_ids']}.", flush=True)


if __name__ == "__main__":
    main()
