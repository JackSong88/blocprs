"""Fit a genome-wide single-source or MultiPRS comparator on the training split."""

import argparse
from pathlib import Path

from blocprs import load_dataset
from blocprs.baseline_model import MultiPRS, SingleSourcePRS


# Read the dataset, comparator and compact set of fitting options.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Packaged genome-wide dataset directory.")
    parser.add_argument("--train-ids", type=Path, help="Training-subset IDs; participants must belong to the stored train split.")
    parser.add_argument("--family", choices=("gaussian", "binomial"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="One fitted model archive (.npz).")
    parser.add_argument("--source", help="Fit this single source instead of MultiPRS; no automatic source selection.")
    parser.add_argument("--variant", choices=("unregularized", "ridge", "lasso", "elasticnet", "nonnegative"),
                        default="unregularized", help="MultiPRS variant (default: unregularized).")
    parser.add_argument("--lambda", dest="lambda_", type=float, help="Fixed absolute penalty; omit for the variant's CV grid.")
    parser.add_argument("--l1-ratio", type=float, help="One elastic-net L1 ratio; omit for the default ratio/grid.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-max-samples", type=int, default=50000, help="Total participants available to CV, not per fold.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


# Fit the stored training split and save the coefficients and source scaling.
def main(argv=None):
    args = parse_args(argv)
    data = load_dataset(args.dataset, split="train", ids_file=args.train_ids)

    if args.source:
        if args.variant != "unregularized" or args.lambda_ is not None or args.l1_ratio is not None:
            raise ValueError("--source uses unregularized fitting; do not supply MultiPRS penalty options.")

        model = SingleSourcePRS(args.source, args.family)
    else:
        model = MultiPRS(args.family, variant=args.variant, lambda_=args.lambda_,
                         l1_ratios=None if args.l1_ratio is None else [args.l1_ratio],
                         cv_folds=args.cv_folds, cv_max_samples=args.cv_max_samples,
                         random_state=args.seed)

    model.fit(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"N={model.n_training_samples_} method={model.method_name_} "
          f"lambda={model.selected_lambda_:g} saved={args.output}")


if __name__ == "__main__":
    main()
