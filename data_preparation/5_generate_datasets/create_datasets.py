# Adapts MoEPRS for BLocPRS: https://github.com/shz9/moe-prs-paper

"""Aligns PRSs and clinical records by participant ID, applies saved splits and saves arrays."""

import argparse
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd
from blocprs.selection import read_ids

from dataset_utils import ScoreTable, participant_index, read_metadata, feature_layout, match_score_participants

logger = logging.getLogger(__name__)


def read_split_ids(path):
    table = read_ids(path)
    keys = table.iloc[:, 0].tolist() if table.shape[1] == 1 else list(table.itertuples(index=False, name=None))

    return keys, table.shape[1]


# Uses the saved training/test order, or keeps all eligible transfer participants.
def assign_splits(metadata, family, train_ids_file=None, test_ids_file=None, all_participants=False):

    if (train_ids_file is None) != (test_ids_file is None):
        raise ValueError("Supply both train and test ID files together.")

    if all_participants and train_ids_file is not None:
        raise ValueError("Choose saved split files or all participants, not both.")

    # Keeps all eligible transfer participants without a new split.
    if all_participants:
        result = metadata.copy()
        result["split"] = "all"

        return result

    train_ids, train_width = read_split_ids(train_ids_file)
    test_ids, test_width = read_split_ids(test_ids_file)

    if train_width != test_width or not set(train_ids).isdisjoint(test_ids):
        raise ValueError("Train/test files must use the same ID format and have no overlap.")

    index = metadata.index if train_width == 2 else pd.Index(metadata.index.get_level_values("IID"))

    if index.has_duplicates:
        raise ValueError("IIDs are not unique; supply two-column FID/IID split files.")

    if not set(index).issubset(set(train_ids).union(test_ids)):
        raise ValueError("Some eligible participants are absent from both split files.")

    # Checks for missing IDs (-1) before using the positions as array indices.
    train_rows, test_rows = index.get_indexer(train_ids), index.get_indexer(test_ids)
    missing = int((train_rows < 0).sum() + (test_rows < 0).sum())

    if missing:
        raise ValueError(f"{missing} saved split IDs are absent from the eligible dataset; check inputs and eligibility.")

    if family == "binomial" and set(metadata.iloc[train_rows]["phenotype"]) != {0, 1}:
        raise ValueError("Binary training data must contain both cases and controls.")

    result = metadata.iloc[np.concatenate([train_rows, test_rows])].copy()
    result["split"] = ["train"] * len(train_rows) + ["test"] * len(test_rows)

    return result


# Reads feature blocks and reorders them into participant rows.
def write_feature_ordered_scores(table, metadata, features, output_dir):
    columns = features["column"].tolist()
    column_positions = {name: index for index, name in enumerate(columns)}
    rows = participant_index(table.ids, "Scores").get_indexer(metadata.index)

    if (rows < 0).any():
        raise ValueError("Selected participants are absent from the input scores.")

    shape = (len(metadata), features["window"].nunique(), features["source"].nunique())
    started = last_log = monotonic()
    written = np.zeros(len(columns), dtype=bool)

    # Reorders scores in temporary storage before one sequential copy to the output.
    with TemporaryDirectory(prefix="blocprs_dataset_", dir=os.environ.get("SLURM_TMPDIR")) as work:
        path = Path(work) / "scores.npy"
        logger.info("Reordering scores in temporary storage: N=%s, features=%s",
                    len(metadata), len(columns))

        scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        matrix = scores.reshape(len(metadata), -1)

        try:
            for names, values in table.iter_feature_chunks():

                if not np.isfinite(values).all():
                    raise ValueError("Scores must be finite and representable as float32.")

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


# Chooses feature-wise or row-wise reads to match the input array layout.
def write_arrays(table, metadata, features, covariate_names, output_dir):
    n_windows = features["window"].nunique()

    if table.array.flags.f_contiguous and n_windows > 1:
        write_feature_ordered_scores(table, metadata, features, output_dir)
    else:
        write_row_ordered_scores(table, metadata, features, output_dir)

    # Saves covariates and labels in the same order as the score rows.
    covariates = metadata[covariate_names].to_numpy(dtype=np.float32)

    if not np.isfinite(covariates).all():
        raise ValueError("Covariates must be finite and representable as float32.")

    np.save(output_dir / "covariates.npy", covariates, allow_pickle=False)
    metadata[["phenotype", "Ancestry", "split"]].reset_index().to_csv(output_dir / "participants.tsv", sep="\t", index=False)
    features.to_csv(output_dir / "features.tsv", sep="\t", index=False)
    pd.DataFrame({"covariate": covariate_names}).to_csv(output_dir / "covariates.tsv", sep="\t", index=False)


# Reads small participant blocks when scores are already stored in that order.
def write_row_ordered_scores(table, metadata, features, output_dir):
    columns = features["column"].tolist()
    n_windows, n_sources = features["window"].nunique(), features["source"].nunique()
    scores = np.lib.format.open_memmap(output_dir / "scores.npy", mode="w+", dtype=np.float32,
                                     shape=(len(metadata), n_windows, n_sources))
    written = np.zeros(len(metadata), dtype=bool)
    chunks = table.iter_chunks(columns=columns, chunk_size=10000, dtype=np.float32)
    started = last_log = monotonic()
    completed = 0
    logger.info("Writing scores: N=%s, windows=%s, sources=%s", len(metadata), n_windows, n_sources)

    try:
        for block, (chunk_ids, values) in enumerate(chunks, start=1):
            ids = participant_index(chunk_ids, "Scores")

            if not np.isfinite(values).all():
                raise ValueError("Scores must be finite and representable as float32.")

            # Maps streamed rows into the saved train/test order, not score-file order.
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

            # Flushes completed blocks so dirty mapped pages do not accumulate.
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

    assert family in ("gaussian", "binomial")
    assert num_pcs >= 0

    if not all_participants and (train_ids_file is None or test_ids_file is None):
        raise ValueError("Supply train/test ID files from create_splits.py, or use --all-participants for transfer data.")

    output_dir = Path(output_dir)

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing dataset: {output_dir}")

    started = monotonic()
    logger.info("Preparing dataset: scores=%s; family=%s", scores_file, family)

    # Aligns clinical records with the available score participants.
    table = ScoreTable(scores_file)
    features = feature_layout(table)
    metadata, covariate_names = read_metadata(phenotype_file, covariates_file, ancestry_file, num_pcs, family)
    metadata, score_count, missing_scores = match_score_participants(table, metadata)

    # Reuses the same participant membership and order for every score representation.
    metadata = assign_splits(metadata, family, train_ids_file, test_ids_file, all_participants)

    logger.info("Selected N=%s; splits=%s; windows=%s; sources=%s", len(metadata),
                metadata["split"].value_counts().to_dict(), features["window"].nunique(), features["source"].nunique())

    # Finishes the arrays in a temporary directory before saving the dataset.
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output_dir.parent, prefix=".dataset_") as temporary:
        temporary = Path(temporary)
        write_arrays(table, metadata, features, covariate_names, temporary)

        # Saves completed files without replacing an existing dataset.
        output_dir.mkdir()
        saved_files = []

        try:
            for path in temporary.iterdir():
                destination = output_dir / path.name
                os.link(path, destination)
                saved_files.append(destination)
        except BaseException:
            for path in saved_files:
                path.unlink()
            output_dir.rmdir()
            raise

    logger.info("Dataset saved: %s; elapsed=%.1fs", output_dir, monotonic() - started)

    return {"participants": len(metadata), "score_rows": score_count, "eligible_without_scores": missing_scores,
            "splits": metadata["split"].value_counts().to_dict()}


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-file", required=True, help="Directory containing scores.npy, participants.tsv and features.tsv.")
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
    print(f"Score participants: {result['score_rows']}; otherwise eligible participants without scores: {result['eligible_without_scores']}.", flush=True)


if __name__ == "__main__":
    main()
