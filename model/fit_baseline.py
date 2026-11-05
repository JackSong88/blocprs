"""Fit a genome-wide single-source or MultiPRS comparator on the training split."""

import argparse
import logging
from time import monotonic
from pathlib import Path

from blocprs import load_dataset
from blocprs.baseline_model import MultiPRS, SingleSourcePRS


# Parses dataset paths, comparator choices and fitting options.
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


# Fits the stored training split and saves the coefficients and source scaling.
def main(argv=None):
    started = monotonic()
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Loads raw S_tilde_ik for the stored training split or its requested subset.
    logging.info("Loading training dataset: %s", args.dataset)
    data = load_dataset(args.dataset, split="train", ids_file=args.train_ids)
    logging.info("Fitting %s: family=%s, N=%s, sources=%s",
                 args.source or args.variant, args.family, data.n_samples, len(data.source_names))

    if args.source:
        assert args.variant == "unregularized" and args.lambda_ is None and args.l1_ratio is None

        model = SingleSourcePRS(args.source, args.family)
    else:
        model = MultiPRS(args.family, variant=args.variant, lambda_=args.lambda_,
                         l1_ratios=None if args.l1_ratio is None else [args.l1_ratio],
                         cv_folds=args.cv_folds, cv_max_samples=args.cv_max_samples,
                         random_state=args.seed)

    # Fits gamma_k and covariate effects, then saves them with training mu_k and sigma_k.
    model.fit(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    logging.info("Saved model: %s; elapsed=%.1fs", args.output, monotonic() - started)
    print(f"N={model.n_training_samples_} method={model.method_name_} "
          f"lambda={model.selected_lambda_:g} saved={args.output}")


if __name__ == "__main__":
    main()
