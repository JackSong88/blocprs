"""Creates one training/test split for genome-wide and windowed PRS datasets.

Requires the score inputs to cover the same participants with complete phenotype
and covariate records. Saves train_ids.txt and test_ids.txt so dataset creation
uses the same people and split order for both score representations.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from dataset_utils import ScoreTable, read_metadata, match_score_participants


# Checks that score inputs cover the same eligible participants before splitting.
def create_splits(scores_files, phenotype_file, covariates_file, ancestry_file, output_dir, family,
                  num_pcs=10, test_fraction=0.3, seed=7209):

    assert family in ("gaussian", "binomial")
    assert num_pcs >= 0
    assert scores_files

    # Selects complete phenotype/covariate records before matching the score IDs.
    metadata, _ = read_metadata(phenotype_file, covariates_file, ancestry_file, num_pcs, family)
    eligible = None
    missing_scores = 0

    for path in scores_files:
        selected, _, missing = match_score_participants(ScoreTable(path), metadata)

        # Sorts IDs so input-file order cannot change the random split.
        selected = selected.sort_index()

        # Requires identical eligible IDs rather than silently dropping people from one input.
        if eligible is None:
            eligible, missing_scores = selected, missing
        elif not selected.index.equals(eligible.index):
            raise ValueError("Score inputs cover different eligible participants; align them before creating splits.")

    # Preserves case/control proportions for binary phenotypes.
    labels = eligible["phenotype"] if family == "binomial" else None
    train_rows, test_rows = train_test_split(np.arange(len(eligible)), test_size=test_fraction,
                                            stratify=labels, random_state=seed)

    if family == "binomial" and set(eligible.iloc[train_rows]["phenotype"]) != {0, 1}:
        raise ValueError("Binary training data must contain both cases and controls.")

    # Groups training and test IDs for the two reusable split files.
    split = eligible.iloc[np.concatenate([train_rows, test_rows])].copy()
    split["split"] = ["train"] * len(train_rows) + ["test"] * len(test_rows)
    output_dir = Path(output_dir)

    # Creates the output directory and fails if it already exists.
    output_dir.mkdir(parents=True, exist_ok=False)

    for name in ("train", "test"):
        ids = split.loc[split["split"].eq(name)].reset_index()[["FID", "IID"]]
        ids.to_csv(output_dir / f"{name}_ids.txt", sep="\t", index=False)

    return {"participants": len(split), "eligible_without_scores": missing_scores,
            "splits": split["split"].value_counts().to_dict()}


# Writes separate training and test ID files.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-files", nargs="+", required=True,
                        help="Score inputs sharing the split, e.g. genome-wide and windowed score directories.")
    parser.add_argument("--phenotype-file", required=True, help="Headerless FID, IID, phenotype table.")
    parser.add_argument("--covariates-file", required=True, help="Headerless FID, IID, Sex, PC1..PCN, Age table.")
    parser.add_argument("--ancestry-file", required=True, help="TSV with FID, IID and Ancestry headers.")
    parser.add_argument("--output-dir", required=True, help="New directory for train_ids.txt and test_ids.txt.")
    parser.add_argument("--family", required=True, choices=("gaussian", "binomial"))
    parser.add_argument("--num-pcs", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7209)
    args = parser.parse_args()
    print("Aligning scores, phenotypes and covariates before creating one shared split.", flush=True)

    try:
        result = create_splits(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Saved splits for {result['participants']} participants to {args.output_dir}: {result['splits']}.")
    print(f"Otherwise eligible participants without scores: {result['eligible_without_scores']}.")


if __name__ == "__main__":
    main()
