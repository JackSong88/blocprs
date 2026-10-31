"""Evaluate fixed genetic scores using covariate-adjusted R^2."""

import argparse
import csv
import gzip
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from blocprs import BLocPRS, FixedCompositionBLocPRS, load_dataset, load_model
from blocprs.data import select_sources
from blocprs.metrics import IncrementalR2, NagelkerkeR2, NonEstimableError
from blocprs.selection import read_participants

if __package__:
    from .bootstrap import bootstrap_metrics, paired_comparisons
    from .discrimination import evaluate_auroc
else:
    from bootstrap import bootstrap_metrics, paired_comparisons
    from discrimination import evaluate_auroc


# Align external scores to the selected participants without dropping anyone.
def load_scores(path, sample_ids, columns=None, phenotype=None):
    path = Path(path)
    open_table = gzip.open if path.suffix == ".gz" else open

    # Check headers before pandas renames duplicate columns.
    with open_table(path, "rt", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"))

    if len(header) != len(set(header)) or any(not name.strip() for name in header):
        raise ValueError("Score columns must have unique, nonempty names.")

    table = pd.read_csv(path, sep="\t", dtype={"FID": str, "IID": str}, keep_default_na=False,
                        float_precision="round_trip")
    ids = table[["FID", "IID"]]

    if any(ids[name].str.strip().eq("").any() for name in ids):
        raise ValueError("Score IDs must not be empty.")

    if columns is None and "phenotype" in table and phenotype is None:
        raise ValueError("The score table contains phenotype metadata; select genetic columns with --score-columns.")

    columns = [name for name in table if name not in ("FID", "IID", "phenotype")] if columns is None else columns

    if not columns or len(columns) != len(set(columns)) or set(columns) & {"FID", "IID", "phenotype"}:
        raise ValueError("Select distinct genetic score columns, not IDs or phenotype metadata.")

    # Duplicate or missing IDs raise errors; unused extra participants are allowed.
    table = table.set_index(["FID", "IID"], verify_integrity=True)
    selected_ids = pd.MultiIndex.from_arrays(sample_ids.T, names=["FID", "IID"])

    # .loc follows the requested ID order and raises for missing rows instead of taking an intersection.
    aligned = table.loc[selected_ids]

    if phenotype is not None and "phenotype" in aligned:

        if not np.array_equal(aligned["phenotype"].to_numpy(dtype=float), phenotype):
            raise ValueError("Prediction phenotypes differ from the evaluation dataset.")

    return aligned[columns].astype(float).reset_index(drop=True)


# Keep sparse groups visible without reporting unstable calibrated estimates.
def reporting_reason(y, family):
    if len(y) < 30:
        return f"Reporting requires at least 30 participants; found {len(y)}."

    if family == "binomial":
        cases, controls = int(np.sum(y == 1)), int(np.sum(y == 0))

        if min(cases, controls) < 10:
            return f"Reporting requires at least 10 cases and 10 controls; found {cases} and {controls}."

    return ""


# Apply the shared metric to every method on exactly the same evaluation rows.
def evaluate_scores(data, scores, family, split):
    y = np.asarray(data.y, dtype=np.float64)
    covariates = np.asarray(data.covariates, dtype=np.float64)
    values = scores.to_numpy(dtype=np.float64)

    # Invalid data must fail, even when a metric would otherwise be non-estimable.
    if y.shape != (data.n_samples,) or not np.isfinite(y).all():
        raise ValueError("Evaluation requires one finite phenotype per participant.")

    if covariates.shape[0] != len(y) or not np.isfinite(covariates).all():
        raise ValueError("Evaluation covariates must be finite and match the participants.")

    if values.shape[0] != len(y) or not np.isfinite(values).all():
        raise ValueError("Every method needs a finite score for every selected participant.")

    if family == "binomial" and not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Binary phenotypes must be coded as 0 and 1.")

    binary = family == "binomial"
    metric_type = NagelkerkeR2 if binary else IncrementalR2
    metric_name = "nagelkerke_r2" if binary else "incremental_r2"
    cases = int(np.count_nonzero(y == 1)) if binary else None
    controls = len(y) - cases if binary else None

    # Reuse the covariate-only fit across all methods.
    metric, null_failure = None, reporting_reason(y, family)
    if not null_failure:
        try:
            metric = metric_type(y, covariates)
        except (NonEstimableError, np.linalg.LinAlgError) as exc:
            null_failure = str(exc) or type(exc).__name__

    rows = []
    for name in scores:
        value, reason = np.nan, null_failure

        if metric is not None:
            try:
                value = metric.score(scores[name].to_numpy(dtype=np.float64))
            except (NonEstimableError, np.linalg.LinAlgError) as exc:
                reason = str(exc) or type(exc).__name__

        rows.append({"method": name, "metric": metric_name, "value": value,
                     "n": len(y), "cases": cases, "controls": controls,
                     "status": "NE" if reason else "OK", "reason": reason, "split": split})

    return pd.DataFrame(rows)


# Select named evaluation groups without changing predictions or participant order.
def group_masks(participants, groups):
    for group in groups:
        mask = np.ones(len(participants), dtype=bool)
        for label in group.split(":"):
            if label == "all":
                continue
            if label in ("female", "male"):
                mask &= participants["Sex"].eq(0 if label == "female" else 1).to_numpy()
            elif label == "non-EUR":
                mask &= participants["Ancestry"].notna().to_numpy() & participants["Ancestry"].ne("EUR").to_numpy()
            else:

                if label not in set(participants["Ancestry"]):
                    raise ValueError(f"Unknown or absent ancestry group: {label}")

                mask &= participants["Ancestry"].eq(label).to_numpy()
        yield group, mask


# Keep method, source and window size separate in the saved results.
def score_labels(columns, window_mb=0):
    rows = []
    for column in columns:
        name, separator, source = column.partition(":")
        match = re.fullmatch(r"(.+)_mb(\d+)", name)
        method, mb = (match.group(1), int(match.group(2))) if match else (name, window_mb)
        method = "single" if method == "source" else method

        if method in ("single", "calibrated_single", "multiprs", "ridge", "lasso", "elastic_net", "nonnegative"):
            mb = 0

        if method in ("blocprs", "fixed_composition", "regional_single", "anchor_only") and mb <= 0:
            raise ValueError(f"Specify the window size for regional score {column!r}.")

        if method in ("single", "regional_single") and not separator:
            raise ValueError(f"Name source scores as {method}:SOURCE, including the source identifier.")

        rows.append(dict(score=column, method=method, source=source, window_mb=mb))
    return pd.DataFrame(rows).set_index("score")


# Compute only the fitted genome-wide component, with the same training scaling.
def anchor_scores(model, data):
    values = np.empty(data.n_samples, dtype=float)
    for start in range(0, data.n_samples, model.block_size):
        rows = slice(start, min(start + model.block_size, data.n_samples))
        scores = model.scaler_.transform_windows(data.window_scores[rows])
        values[rows] = np.einsum("iwk,k->i", scores, model.gamma_, dtype=np.float64)
    return values


# Attach the same scientific labels to point estimates and paired comparisons.
def label_results(table, labels, context):
    if table.empty:
        return table

    table = table.copy()
    names = table["method"].tolist()
    table["score"] = names
    for column in ("method", "source", "window_mb"):
        table[column] = labels.loc[names, column].to_numpy()

    if "reference" in table:
        references = table["reference"].tolist()
        table["reference_score"] = references
        for column in ("method", "source", "window_mb"):
            output = "reference" if column == "method" else f"reference_{column}"
            table[output] = labels.loc[references, column].to_numpy()

    for name, value in context.items():
        table[name] = value
    return table


# Evaluate each requested group using the same rows for every method.
def evaluate_groups(data, participants, scores, family, groups, split, metrics,
                    n_boot=0, seed=20260731, references=()):
    points, comparisons = [], []
    for group, mask in group_masks(participants, groups):
        selected = SimpleNamespace(y=np.asarray(data.y)[mask],
                                   covariates=np.asarray(data.covariates)[mask],
                                   n_samples=int(mask.sum()))
        values = scores.loc[mask].reset_index(drop=True)

        if "r2" in metrics:
            result = evaluate_scores(selected, values, family, split)
            result["group"] = group

            if n_boot:
                try:

                    if result["status"].ne("OK").any():
                        raise NonEstimableError("At least one observed metric is non-estimable.")

                    uncertainty = bootstrap_metrics(selected.y, selected.covariates, values,
                                                    family=family, n_boot=n_boot, seed=seed)
                    result = result.merge(uncertainty.method_summary.drop(columns="observed"),
                                          on="method", validate="one_to_one")
                    result["uncertainty_status"] = "OK"
                    for reference in references:
                        paired = paired_comparisons(uncertainty, reference).rename(columns={
                            "bootstrap_se": "difference_se", "ci_lower": "difference_ci_lower",
                            "ci_upper": "difference_ci_upper"})
                        paired["metric"], paired["group"] = result["metric"].iloc[0], group
                        paired["n"], paired["status"] = selected.n_samples, "OK"
                        paired["uncertainty_method"] = "paired_bootstrap"
                        comparisons.append(paired)
                except NonEstimableError as exc:

                    # Keep failed uncertainty estimates visible; never discard bad resamples.
                    result["uncertainty_status"], result["uncertainty_reason"] = "NE", str(exc)
                    for reference in references:
                        for method in values:
                            if method != reference:
                                comparisons.append(pd.DataFrame([dict(method=method, reference=reference,
                                    metric=result["metric"].iloc[0], group=group, n=selected.n_samples,
                                    status="NE", reason=str(exc), difference=np.nan, difference_se=np.nan,
                                    uncertainty_method="paired_bootstrap")]))
                result["n_boot"], result["seed"] = n_boot, seed
                result["uncertainty_method"] = "paired_bootstrap"
            points.append(result)

        if "auroc" in metrics:
            result, paired = evaluate_auroc(selected.y, values, references)
            reason = reporting_reason(selected.y, family)

            if reason:
                result[["value", "se"]] = np.nan
                result["status"], result["reason"] = "NE", reason
                result["uncertainty_status"] = "NE"

                if not paired.empty:
                    paired[["difference", "difference_se", "p_value"]] = np.nan
                    paired["status"], paired["reason"] = "NE", reason

            result["group"], result["split"] = group, split
            points.append(result)

            if not paired.empty:
                paired["group"] = group
                comparisons.append(paired)

    return pd.concat(points, ignore_index=True), (pd.concat(comparisons, ignore_index=True)
                                                  if comparisons else pd.DataFrame())


# Load fixed genetic scores and write portable point-estimate and comparison tables.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Packaged dataset directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Saved BLocPRS or comparator model archive.")
    source.add_argument("--scores", nargs="+", type=Path, help="Aligned score TSVs; distinct score columns are combined by ID.")
    source.add_argument("--raw-sources", action="store_true", help="Evaluate raw genome-wide input PRSs, without fitted calibration.")
    parser.add_argument("--source-names", nargs="+", help="Named sources to evaluate with --raw-sources; default all.")
    parser.add_argument("--components", action="store_true", help="Also evaluate/export the frozen BLocPRS genome-wide component.")
    parser.add_argument("--score-columns", nargs="+",
                        help="Genetic score columns; required when the input also contains phenotype metadata.")
    parser.add_argument("--family", choices=("gaussian", "binomial"), help="Required for external scores; read from saved models.")
    parser.add_argument("--split", choices=("test", "all"), default="test",
                        help="Stored test split, or the stored all split for transfer evaluation.")
    parser.add_argument("--eval-ids", type=Path, help="Select IDs within the requested evaluation split.")
    parser.add_argument("--groups", nargs="+", default=["all"],
                        help="all, EUR, non-EUR, ancestry labels, female or male; combine as EUR:female.")
    parser.add_argument("--metrics", nargs="+", choices=("r2", "auroc"), default=["r2"])
    parser.add_argument("--phenotype", required=True, help="Analysis identifier, e.g. LDL_MA.")
    parser.add_argument("--cohort", required=True, help="Evaluation cohort, e.g. ukbb or cartagene.")
    parser.add_argument("--train-group", default="all", help="Training population label.")
    parser.add_argument("--train-size", default="full", help="full or the training subset size.")
    parser.add_argument("--window-mb", type=int, default=0, help="Window size for a regional model.")
    parser.add_argument("--panel", default="main", help="Source panel or control condition.")
    parser.add_argument("--covariate-set", default="standard", help="Evaluation covariate-set label.")
    parser.add_argument("--extra-covariates", type=Path, help="FID/IID TSV of additional evaluation covariates; predictions stay fixed.")
    parser.add_argument("--method", help="Optional model score label; external scores use their column names.")
    parser.add_argument("--output", required=True, type=Path, help="Output results TSV.")
    parser.add_argument("--predictions", type=Path, help="Optional TSV containing IDs, phenotype and genetic scores.")
    parser.add_argument("--bootstrap", nargs="?", const=10000, default=0, type=int, metavar="N",
                        help="Add bootstrap SEs and 95%% intervals; default 10000 replicates when enabled.")
    parser.add_argument("--seed", type=int, default=20260731, help="Bootstrap seed (default: 20260731).")
    parser.add_argument("--reference", nargs="+", default=[],
                        help="Score columns for paired bootstrap or DeLong comparisons.")
    args = parser.parse_args()

    if args.bootstrap < 0 or args.bootstrap == 1:
        parser.error("--bootstrap must be 0 (disabled) or at least 2.")

    if args.reference and not args.bootstrap and "auroc" not in args.metrics:
        parser.error("--reference requires --bootstrap or --metrics auroc.")

    if args.model is None and args.family is None:
        parser.error("--family is required with external or raw source scores.")

    if args.scores is None and args.score_columns is not None:
        parser.error("--score-columns applies only to --scores.")

    if args.source_names and not args.raw_sources:
        parser.error("--source-names applies only to --raw-sources.")

    if args.components and args.model is None:
        parser.error("--components requires a saved BLocPRS model.")

    if args.method and args.model is None:
        parser.error("--method applies only to a saved model.")

    if args.score_columns and len(args.scores or []) > 1:
        parser.error("Select columns with one score file, or supply already selected separate files.")

    if args.predictions is not None and args.output.resolve() == args.predictions.resolve():
        parser.error("Results and predictions need different output paths.")

    data = load_dataset(args.dataset, split=args.split, ids_file=args.eval_ids)
    participants = read_participants(args.dataset).set_index(["FID", "IID"])
    participants = participants.loc[pd.MultiIndex.from_arrays(data.sample_ids.T)].reset_index()

    if args.model is not None:
        model = load_model(args.model)

        if args.family is not None and args.family != model.family:
            parser.error("--family differs from the saved model family.")

        if data.covariate_names != model.covariate_names_:
            raise ValueError("Evaluation requires the fitted covariate names and order.")

        family = model.family

        # Keep covariate effects, the intercept and sigmoid out of the PRS.
        data = select_sources(data, model.source_names_)
        if isinstance(model, FixedCompositionBLocPRS):
            name = f"fixed_composition_mb{args.window_mb}"
        elif isinstance(model, BLocPRS):
            name = (f"regional_single_mb{args.window_mb}:{model.source_names_[0]}"
                    if len(model.source_names_) == 1 else f"blocprs_mb{args.window_mb}")
        elif len(model.source_names_) == 1 and model.variant == "unregularized":
            name = f"calibrated_single:{model.source_names_[0]}"
        else:
            name = {"unregularized": "multiprs", "elasticnet": "elastic_net"}.get(model.variant, model.variant)
        name = args.method or name
        scores = pd.DataFrame({name: model.predict_prs(data)})

        if args.components:

            if not isinstance(model, BLocPRS):
                parser.error("--components requires full BLocPRS, not a comparator.")

            scores[f"anchor_only_mb{args.window_mb}"] = anchor_scores(model, data)
    elif args.raw_sources:
        family = args.family

        if data.window_names != ("genomewide",):
            parser.error("--raw-sources requires a genome-wide dataset, not an inferred sum of windows.")

        data = select_sources(data, args.source_names or data.source_names)
        scores = pd.DataFrame(np.asarray(data.window_scores[:, 0], dtype=float),
                              columns=[f"single:{source}" for source in data.source_names])
    else:
        family = args.family
        frames = [load_scores(path, data.sample_ids, args.score_columns, data.y) for path in args.scores]
        scores = pd.concat(frames, axis=1)

    if args.extra_covariates:

        # Predictions have already been computed. Extra covariates change only the evaluation
        # regressions, allowing sensitivity analysis without changing the fitted genetic score.
        extra = load_scores(args.extra_covariates, data.sample_ids, phenotype=data.y)

        if set(extra.columns) & set(data.covariate_names):
            parser.error("Additional covariates must have distinct names from the dataset covariates.")

        data = SimpleNamespace(y=data.y, sample_ids=data.sample_ids, n_samples=data.n_samples,
                               covariates=np.column_stack((data.covariates, extra.to_numpy(dtype=float))))
        if args.covariate_set == "standard":
            args.covariate_set = "expanded"

    if scores.columns.duplicated().any():
        parser.error("Each genetic score column must have a distinct name.")

    if any(reference not in scores for reference in args.reference):
        parser.error("--reference must name evaluated score columns.")

    if "auroc" in args.metrics and family != "binomial":
        parser.error("AUROC requires a binomial phenotype.")

    if len(args.groups) != len(set(args.groups)) or len(args.metrics) != len(set(args.metrics)):
        parser.error("Groups and metrics must not be repeated.")

    labels = score_labels(scores.columns, args.window_mb)
    results, comparisons = evaluate_groups(data, participants, scores, family, args.groups, args.split,
                                           args.metrics, args.bootstrap, args.seed, args.reference)
    context = dict(phenotype=args.phenotype, cohort=args.cohort, train_group=args.train_group,
                   train_size=args.train_size, panel=args.panel, covariate_set=args.covariate_set,
                   family=family, split=args.split)
    results = label_results(results, labels, context)
    comparisons = label_results(comparisons, labels, context)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, sep="\t", index=False, na_rep="NA")

    if not comparisons.empty:
        path = args.output.with_name(args.output.stem + "_comparisons.tsv")
        comparisons.to_csv(path, sep="\t", index=False, na_rep="NA")

    if args.predictions is not None:
        predictions = pd.DataFrame(data.sample_ids, columns=["FID", "IID"])
        predictions["phenotype"] = data.y
        predictions = pd.concat([predictions, scores], axis=1)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions, sep="\t", index=False)

    display = ["group", "method", "metric", "value", "n", "status"]
    display += [name for name in ("bootstrap_se", "difference", "difference_se") if name in results]
    print(results[display].to_string(index=False, na_rep="NA"))


if __name__ == "__main__":
    main()
