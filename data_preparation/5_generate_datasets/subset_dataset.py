"""Write participant-ID subsets without copying PRS arrays or changing saved splits."""

import argparse
from pathlib import Path

from blocprs.selection import filter_participants, id_positions, read_participants
from blocprs.selection import sample_subsets, stratum_labels


# Select a stored population and, optionally, training sample sizes.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for small FID/IID tables.")
    parser.add_argument("--split", choices=("train", "test", "all"), default="train",
                        help="Stored split to filter; sample-size subsets are training-only.")
    parser.add_argument("--ids", type=Path, help="Restrict to these IDs within the selected split.")
    parser.add_argument("--ancestry", nargs="+", help="Recorded ancestry labels, or non-EUR.")
    parser.add_argument("--sex", nargs="+", help="Recorded Sex values, e.g. 0 or 1; coding is not inferred.")
    parser.add_argument("--filter", action="append", default=[], help="Additional COLUMN=VALUE filter; repeat for multiple columns.")
    parser.add_argument("--sizes", nargs="+", type=int, help="One or more exact training sample sizes.")
    parser.add_argument("--nested", action="store_true", help="Smaller subsets are contained in every larger subset.")
    strata = parser.add_mutually_exclusive_group()
    strata.add_argument("--stratify", nargs="+", help="Additional joint strata; binary phenotype is included automatically.")
    strata.add_argument("--no-stratify", action="store_true", help="Sample without stratification, including for binary phenotypes.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


# Keep test membership fixed; omitted training participants never become test rows.
def main(argv=None):
    args = parse_args(argv)

    if args.sizes and args.split != "train":
        raise ValueError("Sample-size subsets apply to the training split; test/all splits may only be filtered.")

    participants = read_participants(args.dataset, split=args.split)
    if args.ids:
        participants = participants.iloc[id_positions(participants, args.ids)]
    participants = filter_participants(participants, args.ancestry, args.sex, args.filter)
    if args.sizes:
        labels = stratum_labels(participants, args.stratify, binary_default=not args.no_stratify)
        subsets = sample_subsets(len(participants), args.sizes, labels, args.nested, args.seed)
        outputs = {f"train_n{size}.ids.tsv": participants.iloc[rows] for size, rows in subsets.items()}
    else:
        outputs = {f"{args.split}.ids.tsv": participants}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        path = args.output_dir / filename
        table[["FID", "IID"]].to_csv(path, sep="\t", index=False)
        print(f"N={len(table)} saved={path}")


if __name__ == "__main__":
    main()
