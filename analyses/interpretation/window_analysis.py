"""Frozen-model window ablations and descriptive training-GWAS concentration.

Window rankings come from training contributions. Ablations change only which
regional terms are retained; neither the PRS nor its coefficients are refitted.
GWAS concentration summarizes completed association tests, not independent
validation or LD-adjusted enrichment.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from blocprs import BLocPRS, load_dataset, load_model
from blocprs.baseline_model import MultiPRS
from blocprs.metrics import IncrementalR2, NagelkerkeR2, NonEstimableError
from evaluation.bootstrap import bootstrap_metrics
from evaluation.evaluate import evaluate_scores

from .contributions import descending_rank, iter_components, validate_context


FRACTIONS = (.01, .025, .05, .10, .25, .50)
GWAS_CAP = 50.0


# Align a complete, one-based training ranking to the fitted window order.
def ranking_order(windows, names):
    ordered = windows.set_index("window", verify_integrity=True)

    if set(ordered.index) != set(names):
        raise ValueError("The training ranking must cover every fitted window exactly once.")

    ranks = ordered.loc[list(names), "adjustment_rank"].to_numpy(dtype=float)

    if not np.array_equal(np.sort(ranks), np.arange(1, len(names) + 1)):
        raise ValueError("Adjustment ranks must be a permutation of 1 through the window count.")

    return np.argsort(ranks, kind="stable")


# Sum a small batch of selected-window masks in bounded participant blocks.
def selected_predictions(local, selections, block_size):
    masks = np.zeros((local.shape[1], len(selections)), dtype=np.float64)
    for column, selected in enumerate(selections):
        masks[selected, column] = 1.0
    predictions = np.empty((local.shape[0], len(selections)), dtype=np.float64)
    for start in range(0, len(local), block_size):
        stop = min(start + block_size, len(local))

        # Each mask column sums one selected window set for the current participants.
        predictions[start:stop] = local[start:stop] @ masks
    return predictions


# Keep every random draw; a non-estimable draw makes its aggregate non-estimable.
def summarize_ablation(metrics):
    rows = []
    for keys, group in metrics.groupby(["mode", "fraction", "n_selected"], sort=False):
        row = dict(zip(("mode", "fraction", "n_selected"), keys))
        row.update(n_total=len(group), n_finite=int(np.isfinite(group.value).sum()),
                   n_recovery=int(np.isfinite(group.recovery).sum()))
        for column in ("n", "n_windows", "metric", "anchor_value", "full_value", "seed"):
            row[column] = group[column].iloc[0]
        for column in ("value", "recovery"):
            values = group[column].to_numpy(dtype=float)
            complete = np.isfinite(values).all()
            quantiles = np.quantile(values, [.025, .25, .5, .75, .975]) if complete else [np.nan] * 5
            row[f"{column}_mean"] = float(values.mean()) if complete else np.nan
            row.update({f"{column}_{name}": value for name, value in
                        zip(("q025", "q25", "median", "q75", "q975"), quantiles)})
        rows.append(row)
    return pd.DataFrame(rows)


# Evaluate centered-raw regional terms without refitting the genetic predictor.
def evaluate_ablation(model, data, windows, *, fractions=FRACTIONS, n_random=1000,
                      seed=42, block_size=1024, work_dir=None):
    fractions = tuple(fractions)

    if (not fractions or len(set(fractions)) != len(fractions)
        or any(not 0 < fraction < 1 for fraction in fractions) or n_random < 0):
        raise ValueError("Use distinct interior fractions and a nonnegative random-draw count.")

    if data.covariate_names != model.covariate_names_:
        raise ValueError("Evaluation covariates must match the fitted covariate names and order.")

    order = ranking_order(windows, model.window_names_)
    metric_name = "nagelkerke_r2" if model.family == "binomial" else "incremental_r2"
    metric_type = NagelkerkeR2 if model.family == "binomial" else IncrementalR2
    metric, null_reason = None, ""
    try:
        metric = metric_type(data.y, data.covariates)
    except (NonEstimableError, np.linalg.LinAlgError) as exc:
        null_reason = str(exc) or type(exc).__name__

    rows = []

    # A metric failure is retained as NE rather than dropping a selected-window set.
    def record(mode, fraction, count, prediction, replicate=-1):
        value, reason = np.nan, null_reason
        if metric is not None:
            try:
                value = metric.score(prediction)
            except (NonEstimableError, np.linalg.LinAlgError) as exc:
                reason = str(exc) or type(exc).__name__
        rows.append(dict(mode=mode, fraction=fraction, n_selected=count, replicate=replicate,
                         metric=metric_name, value=value, n=data.n_samples, n_windows=data.n_windows,
                         status="NE" if reason else "OK", reason=reason, seed=seed))

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Cache only N x W local terms on disk, with no float32 coefficient rounding.
    with TemporaryDirectory(prefix="blocprs_ablation_", dir=work_dir) as temporary:
        local = np.lib.format.open_memmap(Path(temporary) / "local.npy", mode="w+",
                                          dtype=np.float64, shape=(data.n_samples, data.n_windows))
        anchor = np.empty(data.n_samples, dtype=np.float64)
        full_local = np.empty_like(anchor)
        for selection, scores, reference, adjustment in iter_components(model, data, block_size):
            anchor[selection] = reference.sum(axis=1)
            local[selection] = adjustment
            full_local[selection] = adjustment.sum(axis=1)

            # Check reconstruction within the same memory-bounded score block.
            expected = np.einsum("iwk,wk->i", scores, model.c_eff_, dtype=np.float64)
            if not np.allclose(anchor[selection] + full_local[selection], expected, rtol=1e-8, atol=1e-9):
                raise ValueError("Reference and regional terms do not reconstruct the fitted genetic score.")

        local.flush()
        full = anchor + full_local

        record("anchor_only", 0.0, 0, anchor)
        record("full", 1.0, data.n_windows, full)
        random_specs = {}
        for fraction in fractions:
            count = max(1, int(np.ceil(fraction * data.n_windows)))
            prediction = selected_predictions(local, [order[:count]], block_size)[:, 0]
            record("retain_top", fraction, count, anchor + prediction)
            record("drop_top", fraction, count, full - prediction)
            random_specs.setdefault(count, []).append(fraction)

        # Pair retain/drop controls using the same randomly selected windows.
        rng = np.random.default_rng(seed)
        for count, requested_fractions in sorted(random_specs.items()):
            for first in range(0, n_random, 16):
                selections = [np.sort(rng.choice(data.n_windows, size=count, replace=False))
                              for _ in range(first, min(first + 16, n_random))]
                predictions = selected_predictions(local, selections, block_size)
                for column in range(len(selections)):
                    prediction = predictions[:, column]
                    for fraction in requested_fractions:
                        record("retain_random", fraction, count, anchor + prediction, first + column)
                        record("drop_random", fraction, count, full - prediction, first + column)
        del local

    metrics = pd.DataFrame(rows)
    anchor_value, full_value = metrics.value.iloc[:2]
    gain = full_value - anchor_value
    metrics["anchor_value"], metrics["full_value"] = anchor_value, full_value
    metrics["delta_anchor"], metrics["delta_full"] = metrics.value - anchor_value, metrics.value - full_value

    # Negative gains and recovery outside [0, 1] are retained without clipping.
    metrics["recovery"] = metrics.delta_anchor / gain if np.isfinite(gain) and abs(gain) > 1e-12 else np.nan
    summary = summarize_ablation(metrics)
    for table in (metrics, summary):
        table["component_basis"], table["ranking"] = "centered_raw", "training_abs_local"
        table["n_random"] = n_random
    return metrics, summary


# Compare the frozen genome-wide component with an independently fitted MultiPRS.
def evaluate_reference(model, multiprs, data, genomewide, *, n_boot=0, seed=20260731, block_size=1024):
    if type(multiprs) is not MultiPRS or multiprs.variant != "unregularized" or multiprs.family != model.family:
        raise ValueError("The reference comparison requires an unregularized MultiPRS of the same outcome family.")

    if (not np.array_equal(data.sample_ids, genomewide.sample_ids)
        or data.covariate_names != model.covariate_names_
        or data.covariate_names != multiprs.covariate_names_
        or data.covariate_names != genomewide.covariate_names
        or not np.array_equal(data.y, genomewide.y)
        or not np.array_equal(data.covariates, genomewide.covariates)):
        raise ValueError("Windowed and genome-wide evaluation inputs must have identical IDs, outcomes and covariates.")

    anchor = np.empty(data.n_samples, dtype=np.float64)
    for selection, _, reference, _ in iter_components(model, data, block_size):
        anchor[selection] = reference.sum(axis=1)
    scores = pd.DataFrame({"MultiPRS": multiprs.predict_prs(genomewide),
                           "BLocPRS genome-wide component": anchor})
    results = evaluate_scores(data, scores, model.family, "evaluation")
    if n_boot:
        uncertainty = bootstrap_metrics(data.y, data.covariates, scores, family=model.family,
                                        n_boot=n_boot, seed=seed)
        results = results.merge(uncertainty.method_summary.drop(columns="observed"),
                                on="method", validate="one_to_one")
        results["seed"] = seed
    return results


# Validate inclusive genomic intervals before assigning association-test positions.
def genomic_windows(windows):
    result = windows[["window", "CHR", "start_bp", "end_bp"]].copy()
    result["window"] = result.window.astype(str)
    for column in ("CHR", "start_bp", "end_bp"):
        values = pd.to_numeric(result[column], errors="raise")

        if not np.isfinite(values).all() or not values.mod(1).eq(0).all():
            raise ValueError(f"Window coordinate {column} must contain finite integers.")

        result[column] = values.astype(np.int64)

    if result.window.duplicated().any() or result.start_bp.lt(1).any() or result.end_bp.lt(result.start_bp).any():
        raise ValueError("Window IDs must be unique and bounds must be one-based inclusive.")

    result = result.sort_values(["CHR", "start_bp"])
    for _, group in result.groupby("CHR", sort=False):
        if np.any(group.start_bp.to_numpy()[1:] <= group.end_bp.to_numpy()[:-1]):
            raise ValueError("Genomic windows overlap.")
    return result.reset_index(drop=True)


# Retain successful additive tests, with explicit treatment of missing and zero P values.
def gwas_chunks(path, chromosome, counts, chunksize=150000):
    seen = set()
    columns = ["#CHROM", "POS", "ID", "TEST", "P", "ERRCODE"]
    for frame in pd.read_csv(path, sep="\t", usecols=columns, dtype=str,
                             keep_default_na=False, chunksize=chunksize):
        chrom = pd.to_numeric(frame["#CHROM"], errors="raise")
        position = pd.to_numeric(frame.POS, errors="raise")

        if not chrom.eq(chromosome).all() or not (position.ge(1) & position.mod(1).eq(0)).all():
            raise ValueError(f"Unexpected chromosome or position in {path}")

        additive = frame.TEST.eq("ADD")
        ids = frame.loc[additive, "ID"]

        if ids.duplicated().any() or seen.intersection(ids):
            raise ValueError(f"Duplicate additive variant tests in {path}")

        seen.update(ids)
        failed = additive & ~frame.ERRCODE.isin([".", "", "NA"])
        usable = additive & ~failed
        p = pd.to_numeric(frame.P, errors="coerce")

        if (usable & p.notna() & (~np.isfinite(p) | p.lt(0) | p.gt(1))).any():
            raise ValueError(f"Invalid P value in {path}")

        missing = usable & p.isna()

        if (missing & ~frame.P.isin(["NA", "nan", ".", ""])).any():
            raise ValueError(f"Malformed P value in {path}")

        valid = usable & p.notna()
        counts["n_failed_tests"] += int(failed.sum())
        counts["n_missing_p"] += int(missing.sum())
        counts["n_zero_p"] += int((valid & p.eq(0)).sum())
        counts["n_valid_variants"] += int(valid.sum())

        # Zero P denotes numerical underflow; the capped concentration remains defined.
        logp = -np.log10(p.loc[valid].clip(lower=np.nextafter(0.0, 1.0)))
        yield position.loc[valid].to_numpy(dtype=np.int64), logp.to_numpy(dtype=float)


# Describe capped max-logP mass in windows selected by training adjustment ranks.
def compute_gwas_concentration(gwas_files, windows, *, fractions=FRACTIONS):
    coordinates = genomic_windows(windows)
    order = ranking_order(windows, coordinates.window.tolist())

    if set(gwas_files) != set(coordinates.CHR):
        raise ValueError("Supply exactly one GWAS file for every modeled chromosome.")

    tables = []
    counts = dict(n_valid_variants=0, n_unmapped_variants=0, n_failed_tests=0, n_missing_p=0, n_zero_p=0)
    for chromosome, group in coordinates.groupby("CHR", sort=True):
        group = group.copy()
        starts, ends = group.start_bp.to_numpy(), group.end_bp.to_numpy()
        maxima, variants = np.zeros(len(group)), np.zeros(len(group), dtype=np.int64)
        before = counts["n_valid_variants"]
        for positions, logp in gwas_chunks(gwas_files[chromosome], chromosome, counts):
            indices = np.searchsorted(starts, positions, side="right") - 1
            covered = indices >= 0
            covered[covered] &= positions[covered] <= ends[indices[covered]]
            np.add.at(variants, indices[covered], 1)
            np.maximum.at(maxima, indices[covered], logp[covered])
            counts["n_unmapped_variants"] += int((~covered).sum())

        if counts["n_valid_variants"] == before or not variants.sum():
            raise ValueError(f"No successful additive tests map to modeled windows on chromosome {chromosome}.")

        group["n_valid_variants"], group["max_logp"] = variants, maxima
        group["gwas_score"], group["no_valid_variants"] = np.minimum(maxima, GWAS_CAP), variants == 0
        tables.append(group)
    scored = pd.concat(tables, ignore_index=True)
    scores = scored.gwas_score.to_numpy()
    total = float(scores.sum())

    if not np.isfinite(total) or total <= 0:
        raise ValueError("The modeled windows contain no positive GWAS signal mass.")

    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("GWAS concentration fractions must be in (0, 1].")

        # Normalize by the realized window fraction after rounding up the selected count.
        count = max(1, int(np.ceil(len(scored) * fraction)))
        share = float(scores[order[:count]].sum() / total)
        rows.append(dict(fraction=fraction, n_selected=count, n_windows=len(scored),
                         window_share=count / len(scored), signal_share=share,
                         concentration_ratio=share / (count / len(scored)), cap=GWAS_CAP, **counts))
    return scored, pd.DataFrame(rows)


# Verify saved ranks against the matching training-derived coefficient contributions.
def load_context(directory, model=None):
    context = pd.read_csv(directory / "contributions.tsv", sep="\t", dtype={"window": str, "base": str})
    windows = pd.read_csv(directory / "windows.tsv", sep="\t", dtype={"window": str})

    if (context.duplicated(["window", "base"]).any()
        or not np.isfinite(context[["eta_centered", "model_prs_sd"]]).all().all()
        or context.model_prs_sd.lt(0).any()):
        raise ValueError("Training contributions must have unique source/window rows and finite coefficients/SDs.")

    if model is not None:
        context = validate_context(model, context)
        names = list(model.window_names_)
    else:
        names = context.window.drop_duplicates().tolist()
    activity = (context.eta_centered.abs() * context.model_prs_sd).groupby(context.window, sort=False).sum()
    ordered = windows.set_index("window", verify_integrity=True).loc[names]

    if (set(activity.index) != set(windows.window)
        or not np.array_equal(descending_rank(activity.loc[names]), ordered.adjustment_rank.to_numpy())):
        raise ValueError("Window ranks differ from the saved training contribution magnitudes.")

    return context, windows


# Native evaluation must be disjoint from the stored fitting participants.
def check_native_split(training_dataset, data, model):
    participants = pd.read_csv(training_dataset / "participants.tsv", sep="\t",
                               usecols=["FID", "IID", "split"], dtype=str)
    train = participants.loc[participants.split.eq("train"), ["FID", "IID"]]

    if len(train) != model.n_training_samples_ or train.isna().any().any() or train.duplicated().any():
        raise ValueError("The training ID table must match the fitted sample count and contain unique complete IDs.")

    train_ids = pd.MultiIndex.from_frame(train)
    if pd.MultiIndex.from_arrays(data.sample_ids.T).isin(train_ids).any():
        raise ValueError("Native evaluation participants overlap the model training split.")


# Run one phenotype at a time, keeping analysis tables together under its directory.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=("ablation", "reference", "gwas"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analyses/interpretation"))
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--model", type=Path, help="Fitted BLocPRS model; required for ablation.")
    parser.add_argument("--dataset", type=Path, help="Evaluation dataset; required for ablation.")
    parser.add_argument("--training-dataset", type=Path, help="Required for native train/test ID separation.")
    parser.add_argument("--scenario", choices=("ukbb_native", "full_cag_transfer"), default="ukbb_native")
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=Path, default=Path("data/tmp"))
    parser.add_argument("--multiprs-model", type=Path, help="Also compare the frozen genome-wide component with MultiPRS.")
    parser.add_argument("--genomewide-dataset", type=Path, help="Matched genome-wide evaluation data for --multiprs-model.")
    parser.add_argument("--bootstrap", nargs="?", const=10000, default=0, type=int, metavar="N")
    parser.add_argument("--gwas-template", help="Completed GRCh37 PLINK GWAS files with a {chrom} placeholder.")
    args = parser.parse_args()
    directory = args.analysis_dir / args.phenotype

    if args.bootstrap < 0 or args.bootstrap == 1 or (args.bootstrap and args.multiprs_model is None):
        parser.error("--bootstrap requires --multiprs-model and at least two replicates.")

    if (args.multiprs_model is None) != (args.genomewide_dataset is None):
        parser.error("Provide --multiprs-model and --genomewide-dataset together.")

    if args.analysis in ("ablation", "reference"):
        if args.model is None or args.dataset is None:
            parser.error("Ablation needs --model and --dataset.")
        if args.split != ("test" if args.scenario == "ukbb_native" else "all"):
            parser.error("Use the test split for native evaluation and the all split for transfer.")
        if args.scenario == "ukbb_native" and args.training_dataset is None:
            parser.error("Native ablation needs --training-dataset.")
        if args.analysis == "reference" and args.multiprs_model is None:
            parser.error("Reference comparison needs --multiprs-model and --genomewide-dataset.")

        model, data = BLocPRS.load(args.model), load_dataset(args.dataset, split=args.split)
        context, windows = load_context(directory, model)
        if args.scenario == "ukbb_native":
            check_native_split(args.training_dataset, data, model)
        tables = {}
        if args.analysis == "ablation":
            metrics, summary = evaluate_ablation(model, data, windows, n_random=args.n_random,
                                                 seed=args.seed, work_dir=args.work_dir)
            tables.update({"metrics.tsv": metrics, "summary.tsv": summary})
        if args.multiprs_model is not None:
            genomewide = load_dataset(args.genomewide_dataset, split=args.split)
            tables["reference.tsv"] = evaluate_reference(model, load_model(args.multiprs_model), data,
                                                           genomewide, n_boot=args.bootstrap).assign(split=args.split)
        output = directory / "ablation" / args.scenario
        metadata = dict(phenotype=args.phenotype, family="binary" if model.family == "binomial" else "continuous",
                        scenario=args.scenario)
    else:
        if args.gwas_template is None or "{chrom}" not in args.gwas_template:
            parser.error("GWAS concentration needs --gwas-template containing {chrom}.")

        context, windows = load_context(directory)
        files = {int(chrom): Path(args.gwas_template.format(chrom=int(chrom))) for chrom in windows.CHR.unique()}
        scored, concentration = compute_gwas_concentration(files, windows)
        tables = {"windows.tsv": scored, "concentration.tsv": concentration}
        output = directory / "gwas"
        metadata = dict(phenotype=args.phenotype, family=context.family.iloc[0])

    if not context.phenotype.eq(args.phenotype).all() or not windows.phenotype.eq(args.phenotype).all():
        raise ValueError("Contribution inputs name a different phenotype.")

    if "window_size" in windows:
        sizes = windows.window_size.drop_duplicates()
        if len(sizes) != 1:
            raise ValueError("Analysis inputs must describe one window size.")
        metadata["window_size"] = sizes.iloc[0]

    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.assign(**metadata).to_csv(output / name, sep="\t", index=False, na_rep="NA")
    print(f"{args.phenotype}: saved {args.analysis} tables to {output}")


if __name__ == "__main__":
    main()
