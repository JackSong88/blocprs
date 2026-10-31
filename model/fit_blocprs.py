"""Fit BLocPRS on the training split and save one model archive."""

import argparse
import logging
from pathlib import Path

from blocprs import BLocPRS, load_dataset, select_source
from blocprs.fixed_composition import FixedCompositionBLocPRS


# Read the dataset, output path and fitting options.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Packaged dataset directory.")
    parser.add_argument("--train-ids", type=Path, help="Training-subset IDs; participants must belong to the stored train split.")
    parser.add_argument("--source", help="Fit only this named source, for regional single-source recalibration.")
    parser.add_argument("--model", choices=("blocprs", "fixed-composition"), default="blocprs",
                        help="Full BLocPRS or MultiPRS followed by one-source regional scaling.")
    parser.add_argument("--genomewide-dataset", type=Path,
                        help="Required for fixed-composition fitting: aligned, independently scored genome-wide training data.")
    parser.add_argument("--family", choices=("gaussian", "binomial"), required=True,
                        help="Gaussian for continuous phenotypes; binomial for 0/1 phenotypes.")
    parser.add_argument("--output", type=Path, required=True, help="Fitted model archive, e.g. model.npz.")
    parser.add_argument("--lambda", dest="lambda_", type=float,
                        help="Fixed regional penalty; omit to select from 25 values from 1e-3 to 1e3.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5).")
    parser.add_argument("--cv-workers", type=int, default=1,
                        help="Concurrent BLocPRS CV folds; allocate one CPU per worker (default: 1).")
    parser.add_argument("--fit-workers", type=int, default=1,
                        help="CPUs for the final BLocPRS fit (default: 1).")
    parser.add_argument("--cache-dir", type=Path,
                        help="Temporary model caches; defaults to SLURM_TMPDIR or the system temporary directory.")
    parser.add_argument("--cv-max-samples", type=int, default=50000,
                        help="Maximum total participants used across CV folds (default: 50000).")
    parser.add_argument("--seed", type=int, default=42, help="CV sampling and fold seed (default: 42).")
    parser.add_argument("--block-size", type=int, default=8192,
                        help="Participants processed per design block (default: 8192).")
    return parser.parse_args(argv)


# Fit only the stored training split, then save all fitted parameters together.
def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.model == "fixed-composition" and args.cv_workers != 1:
        raise ValueError("Fixed-composition CV currently requires --cv-workers 1.")

    if args.model == "fixed-composition" and (args.fit_workers != 1 or args.cache_dir is not None):
        raise ValueError("--fit-workers and --cache-dir currently apply to full BLocPRS fitting.")

    data = load_dataset(args.dataset, split="train", ids_file=args.train_ids)
    if args.source:
        data = select_source(data, args.source)

    if (args.model == "fixed-composition") != (args.genomewide_dataset is not None):
        raise ValueError("Use --genomewide-dataset with --model fixed-composition only.")

    model_class = FixedCompositionBLocPRS if args.model == "fixed-composition" else BLocPRS
    options = ({"cv_workers": args.cv_workers, "fit_workers": args.fit_workers, "cache_dir": args.cache_dir}
               if args.model == "blocprs" else {})
    model = model_class(family=args.family, lambda_=args.lambda_, cv_folds=args.cv_folds,
                        cv_max_samples=args.cv_max_samples, random_state=args.seed,
                        block_size=args.block_size, **options)

    if args.model == "fixed-composition":

        # Both stages use the same requested training IDs; the model checks their order and values.
        genomewide = load_dataset(args.genomewide_dataset, split="train", ids_file=args.train_ids)
        if args.source:
            genomewide = select_source(genomewide, args.source)
        model.fit(data, genomewide)
    else:
        model.fit(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"model={args.model} N={model.n_training_samples_} family={model.family} "
          f"lambda={model.selected_lambda_:g} saved={args.output}")


if __name__ == "__main__":
    main()
