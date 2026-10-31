"""Create one participant split for genome-wide and windowed PRS datasets."""

import argparse
from pathlib import Path

from create_datasets import assign_splits, read_metadata, score_participants


# Check score coverage before splitting; only participant IDs are read from the scores.
def create_splits(scores_files, phenotype_file, covariates_file, ancestry_file, output_dir, family,
                  num_pcs=10, test_fraction=0.3, seed=7209, stratify=None):

    if family not in ("gaussian", "binomial") or num_pcs < 0:
        raise ValueError("Use gaussian/binomial family and a nonnegative number of PCs.")

    if not scores_files:
        raise ValueError("Supply the score inputs that will share this split.")

    metadata, _ = read_metadata(phenotype_file, covariates_file, ancestry_file, num_pcs, family)
    eligible = None
    missing_scores = 0
    for path in scores_files:
        selected, _, missing = score_participants(path, metadata)

        # Sort IDs so input-file order cannot change the random split.
        selected = selected.sort_index()
        if eligible is None:
            eligible, missing_scores = selected, missing
        elif not selected.index.equals(eligible.index):
            raise ValueError("Score inputs cover different eligible participants; align them before creating splits.")

    split, _ = assign_splits(eligible, family, test_fraction=test_fraction, seed=seed, stratify=stratify)
    output_dir = Path(output_dir)

    # Keep existing splits intact; generate new splits in a new directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("train", "test"):
        ids = split.loc[split["split"].eq(name)].reset_index()[["FID", "IID"]]
        ids.to_csv(output_dir / f"{name}_ids.txt", sep="\t", index=False)

    return {"participants": len(split), "eligible_without_scores": missing_scores,
            "splits": split["split"].value_counts().to_dict()}


# Save IDs once; pass these files to every subsequent dataset build.
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
    parser.add_argument("--stratify", nargs="+", help="Additional strata, e.g. Ancestry Sex; binary case/control status is included automatically.")
    parser.add_argument("--seed", type=int, default=7209)
    args = parser.parse_args()
    try:
        result = create_splits(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved splits for {result['participants']} participants to {args.output_dir}: {result['splits']}.")
    print(f"Otherwise eligible participants without scores: {result['eligible_without_scores']}.")


if __name__ == "__main__":
    main()
