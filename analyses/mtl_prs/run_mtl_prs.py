"""Run the individual-level MTL-PRS benchmark from packaged cohort inputs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from blocprs import load_dataset
from evaluation.bootstrap import bootstrap_metrics
from evaluation.evaluate import evaluate_scores, load_scores

from .fit import calibrate_baseline, fit_chromosome
from .mixture import select_mixture
from .prepare import prepare_data, prepare_weights
from .score import check_baseline, combine_scores, export_weights, score_chromosome, select_candidate


chromosomes = tuple(range(1, 23))
comparators = ("Fixed-composition regional MultiPRS", "BLocPRS (1 Mb)")
stages = ("prepare", "mixture", "weights", "fit", "score-validation", "select", "score-test", "collect")


# Read a small numerical archive without Python object deserialization.
def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


# Construct participant pairs in the saved dataset order.
def participant_ids(prepared, split):
    if split == "test":
        return np.column_stack((prepared["test_fid"], prepared["test_iid"]))

    rows = prepared[f"{split}_rows"]
    return np.column_stack((prepared["fid"][rows], prepared["iid"][rows]))


# Reserve the inner validation set once, keeping outer test outcomes out of preparation.
def prepare_run(args):
    if args.dataset is None or args.family is None or not args.phenotype:
        raise ValueError("The prepare stage needs --dataset, --family and --phenotype.")

    args.ld_blocks.stat()
    prepared = prepare_data(args.dataset, args.family, args.validation_size, args.seed, args.validation_ids)
    prepared.update(phenotype=np.asarray(args.phenotype),
                    genotype_template=np.asarray(str(Path(args.genotype_template).resolve())),
                    scorefile_template=np.asarray(str(Path(args.scorefile_template).resolve())),
                    ld_blocks=np.asarray(str(args.ld_blocks.resolve())))

    # A new run directory prevents mixing inputs from different experiments.
    args.run_dir.mkdir(parents=True, exist_ok=False)
    np.savez(args.run_dir / "prepared.npz", **prepared)
    print(f"Prepared {len(prepared['update_rows']):,} fitting, "
          f"{len(prepared['validation_rows']):,} validation and {len(prepared['test_iid']):,} test participants.")


# Select raw-score mixture weights on update-training rows, then calibrate that mixture.
def fit_mixture(run, prepared):
    rows = prepared["update_rows"]
    y, covariates, scores = (prepared[name][rows] for name in ("y", "covariates", "scores"))
    pi, candidates = select_mixture(y, scores, covariates, prepared["source_names"],
                                    family=str(prepared["family"].item()), folds=5, step=.05,
                                    seed=int(prepared["seed"].item()))
    calibration = calibrate_baseline(y, covariates, scores, pi)
    candidates.to_csv(run / "mixture.tsv", sep="\t", index=False)
    np.savez(run / "calibration.npz", pi=pi, **calibration)
    print("Selected raw mixture: " + ", ".join(
        f"{source}={value:g}" for source, value in zip(prepared["source_names"], pi)))


# Build chromosome mixtures before defining the genome-wide learning-rate denominator M.
def build_weights(run, prepared, calibration):
    counts = []
    for chromosome in chromosomes:
        prefix = str(prepared["genotype_template"].item()).format(chrom=chromosome)
        count = prepare_weights(chromosome, prefix, str(prepared["scorefile_template"].item()),
                                prepared["source_names"], calibration["pi"],
                                run / "weights" / f"chr{chromosome}.tsv.gz")
        counts.append({"chromosome": chromosome, "variants": count})
        print(f"chr{chromosome}: {count:,} nonzero mixed SNP weights.")
    pd.DataFrame(counts).to_csv(run / "weights/counts.tsv", sep="\t", index=False)


# Fit only update-training participants, using the same genome-wide M on every chromosome.
def fit_models(run, prepared, calibration, selected_chromosomes):
    counts = pd.read_csv(run / "weights/counts.tsv", sep="\t")

    if counts.chromosome.duplicated().any() or set(counts.chromosome) != set(chromosomes):
        raise ValueError("Weight counts must cover chromosomes 1-22 exactly once.")

    total_variants = int(counts.variants.sum())
    ids = participant_ids(prepared, "update")
    for chromosome in selected_chromosomes:
        prefix = str(prepared["genotype_template"].item()).format(chrom=chromosome)
        summary = fit_chromosome(prefix, run / "weights" / f"chr{chromosome}.tsv.gz", ids,
                                 calibration, total_variants, str(prepared["ld_blocks"].item()),
                                 chromosome, run / "models" / f"chr{chromosome}.npz")
        print(f"chr{chromosome}: fitted {summary['fitted_variants']:,} SNPs; "
              f"excluded {summary['dropped_monomorphic']:,} without training variation.")


# Read the validation-selected candidate rather than selecting again on test data.
def selected_label(run):
    table = pd.read_csv(run / "selection.tsv", sep="\t")
    selected = table.loc[table.selected.eq(True), "candidate_label"]

    if len(selected) != 1:
        raise ValueError("The selection table must identify exactly one validation-selected candidate.")

    return str(selected.iloc[0])


# Score all candidates on validation, but only the chosen candidate on the outer test set.
def score_models(run, prepared, split, selected_chromosomes):
    ids = participant_ids(prepared, split)
    label = selected_label(run) if split == "test" else None
    for chromosome in selected_chromosomes:
        prefix = str(prepared["genotype_template"].item()).format(chrom=chromosome)
        score_chromosome(prefix, run / "models" / f"chr{chromosome}.npz", ids,
                         run / split / f"chr{chromosome}.npz", candidate_label=label)
        print(f"chr{chromosome}: scored {len(ids):,} {split} participants.")


# Require baseline reconstruction before validation can select among all 61 candidates.
def select_model(run, prepared, calibration):
    rows = prepared["validation_rows"]
    paths = [run / "validation" / f"chr{chromosome}.npz" for chromosome in chromosomes]
    labels, scores = combine_scores(paths, participant_ids(prepared, "validation"))
    baseline = int(np.flatnonzero(labels == "baseline").item())
    check = check_baseline(scores[:, baseline], prepared["scores"][rows], calibration["pi"], calibration["theta"])
    selection = select_candidate(prepared["y"][rows], prepared["covariates"][rows], labels, scores,
                                 str(prepared["family"].item()))
    selection.to_csv(run / "selection.tsv", sep="\t", index=False)
    selected = selection.loc[selection.selected].iloc[0]
    print(f"Baseline reconstruction: correlation={check['correlation']:.8g}, "
          f"relative RMSE={check['relative_rmse']:.3g}.")
    print(f"Selected {selected.candidate_label}: validation {selected.metric}={selected.value:.6g}.")


# Accept either a joint score table or the two shared evaluator prediction exports.
def load_comparators(paths, ids, phenotype=None):
    aliases = {comparators[0]: (comparators[0], "fixed_composition_mb1"),
               comparators[1]: (comparators[1], "blocprs_mb1")}
    scores = {}
    for path in paths:
        header = pd.read_csv(path, sep="\t", nrows=0).columns
        for method, names in aliases.items():
            matches = [name for name in names if name in header]

            if len(matches) > 1 or (matches and method in scores):
                raise ValueError(f"Multiple comparison-score columns supplied for {method}.")

            if matches:
                scores[method] = load_scores(path, ids, columns=matches, phenotype=phenotype).iloc[:, 0]

    if set(scores) != set(comparators):
        raise ValueError("Supply fixed-composition and 1-Mb BLocPRS genetic scores for the comparison.")

    return pd.DataFrame(scores)[list(comparators)]


# Evaluate frozen genetic scores and export the selected per-allele effects.
def collect_results(args, prepared):
    if "phenotype" not in prepared:
        raise ValueError("Run the prepare stage with --phenotype before collecting results.")

    run = args.run_dir
    ids = participant_ids(prepared, "test")
    paths = [run / "test" / f"chr{chromosome}.npz" for chromosome in chromosomes]
    labels, values = combine_scores(paths, ids)

    if labels.tolist() != [selected_label(run)]:
        raise ValueError("Test scores do not match the validation-selected candidate.")

    data = load_dataset(str(prepared["dataset"].item()), split="test")

    if not np.array_equal(ids, data.sample_ids):
        raise ValueError("Test participant IDs changed after run preparation.")

    scores = pd.DataFrame({"MTL-PRS": values[:, 0]})
    if args.comparison_scores is not None:
        other = load_comparators(args.comparison_scores, ids, data.y)
        scores = pd.concat([scores, other], axis=1)

    family = str(prepared["family"].item())
    results = evaluate_scores(data, scores, family, "test")
    if args.bootstrap:
        uncertainty = bootstrap_metrics(data.y, data.covariates, scores, family=family,
                                        n_boot=args.bootstrap, seed=20260731)
        results = results.merge(uncertainty.method_summary.drop(columns="observed"),
                                on="method", how="left", validate="one_to_one")
        results["seed"] = 20260731

    results["phenotype"] = str(prepared["phenotype"].item())
    results["family"] = "binary" if family == "binomial" else "continuous"
    predictions = pd.concat([pd.DataFrame(ids, columns=["FID", "IID"]), scores], axis=1)
    predictions.to_csv(run / "predictions.tsv.gz", sep="\t", index=False)
    model_paths = [run / "models" / f"chr{chromosome}.npz" for chromosome in chromosomes]
    export_weights(model_paths, labels[0], run / "weights.tsv.gz")
    results.to_csv(run / "results.tsv", sep="\t", index=False, na_rep="NA")
    print(results[["method", "metric", "value", "n", "status"]].to_string(index=False))


# Keep scientific defaults fixed; expose cohort paths, split size and chromosome scheduling.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=stages)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Stage-5 dataset; required by prepare.")
    parser.add_argument("--phenotype", help="Analysis ID, e.g. LDL_MA; required by prepare and saved for subsequent stages.")
    parser.add_argument("--family", choices=("gaussian", "binomial"), help="Required by prepare.")
    parser.add_argument("--validation-size", type=int, default=50000)
    parser.add_argument("--validation-ids", type=Path, help="Explicit FID/IID validation subset instead of subsampling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--genotype-template", default="data/ukbb/genotypes/chr{chrom}")
    parser.add_argument("--scorefile-template", default=(
        "data/canonical/ukbb_fullchr/chr{chrom}/ukbb_{chrom}_additive_{additive}_fullchr.scorefile.gz"))
    parser.add_argument("--ld-blocks", type=Path, default=Path("data/reference/Berisa.EUR.hg19.bed"))
    parser.add_argument("--chromosome", type=int, choices=chromosomes,
                        help="One chromosome for fit or scoring; otherwise all 22.")
    parser.add_argument("--comparison-scores", type=Path, nargs="+",
                        help="Collect: a joint score table or the two comparator prediction exports.")
    parser.add_argument("--bootstrap", nargs="?", const=10000, default=0, type=int, metavar="N",
                        help="Collect: paired bootstrap SEs and intervals; 10000 replicates when enabled.")
    args = parser.parse_args()

    if args.bootstrap < 0 or args.bootstrap == 1:
        parser.error("--bootstrap must be 0 or at least 2.")

    if args.chromosome is not None and args.stage not in {"fit", "score-validation", "score-test"}:
        parser.error("--chromosome applies only to fit and scoring stages.")

    if args.stage == "prepare":
        prepare_run(args)
        return

    prepared = load_arrays(args.run_dir / "prepared.npz")
    selected_chromosomes = chromosomes if args.chromosome is None else (args.chromosome,)
    if args.stage == "mixture":
        fit_mixture(args.run_dir, prepared)
    elif args.stage in {"weights", "fit", "select"}:
        calibration = load_arrays(args.run_dir / "calibration.npz")
        if args.stage == "weights":
            build_weights(args.run_dir, prepared, calibration)
        elif args.stage == "fit":
            fit_models(args.run_dir, prepared, calibration, selected_chromosomes)
        else:
            select_model(args.run_dir, prepared, calibration)
    elif args.stage.startswith("score-"):
        score_models(args.run_dir, prepared, args.stage.removeprefix("score-"), selected_chromosomes)
    else:
        collect_results(args, prepared)


if __name__ == "__main__":
    main()
