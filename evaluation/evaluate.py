"""Evaluate fixed genetic scores using covariate-adjusted R^2."""

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from blocprs import load_dataset, load_model
from blocprs.metrics import IncrementalR2, NagelkerkeR2, NonEstimableError

if __package__:
    from .bootstrap import bootstrap_metrics, paired_comparisons
else:
    from bootstrap import bootstrap_metrics, paired_comparisons


# Align external scores to the selected participants without dropping anyone.
def load_scores(path, sample_ids, columns=None):
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

    if columns is None and "phenotype" in table:
        raise ValueError("The score table contains phenotype metadata; select genetic columns with --score-columns.")

    columns = [name for name in table if name not in ("FID", "IID")] if columns is None else columns

    if not columns or len(columns) != len(set(columns)) or set(columns) & {"FID", "IID", "phenotype"}:
        raise ValueError("Select distinct genetic score columns, not IDs or phenotype metadata.")

    # Duplicate or missing IDs raise errors; unused extra participants are allowed.
    table = table.set_index(["FID", "IID"], verify_integrity=True)
    selected_ids = pd.MultiIndex.from_arrays(sample_ids.T, names=["FID", "IID"])
    return table.loc[selected_ids, columns].astype(float).reset_index(drop=True)


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
    metric, null_failure = None, ""
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


# Load a saved model or external genetic scores and write a compact results table.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Packaged dataset directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Saved BLocPRS or comparator model archive.")
    source.add_argument("--scores", type=Path, help="TSV or TSV.gz with FID, IID and genetic score columns.")
    parser.add_argument("--score-columns", nargs="+",
                        help="Genetic score columns; required when the input also contains phenotype metadata.")
    parser.add_argument("--family", choices=("gaussian", "binomial"), help="Required for external scores; read from saved models.")
    parser.add_argument("--split", choices=("test", "all"), default="test",
                        help="Stored test split, or the stored all split for transfer evaluation.")
    parser.add_argument("--output", required=True, type=Path, help="Output results TSV.")
    parser.add_argument("--predictions", type=Path, help="Optional TSV containing IDs, phenotype and genetic scores.")
    parser.add_argument("--bootstrap", nargs="?", const=10000, default=0, type=int, metavar="N",
                        help="Add bootstrap SEs and 95%% intervals; default 10000 replicates when enabled.")
    parser.add_argument("--seed", type=int, default=20260731, help="Bootstrap seed (default: 20260731).")
    parser.add_argument("--reference", help="With bootstrap, also report paired differences from this score column.")
    args = parser.parse_args()

    if args.bootstrap < 0 or args.bootstrap == 1:
        parser.error("--bootstrap must be 0 (disabled) or at least 2.")

    if args.reference is not None and not args.bootstrap:
        parser.error("--reference requires --bootstrap.")

    if args.scores is not None and args.family is None:
        parser.error("--family is required with --scores.")

    if args.model is not None and args.score_columns is not None:
        parser.error("--score-columns applies only to --scores.")

    if args.predictions is not None and args.output.resolve() == args.predictions.resolve():
        parser.error("Results and predictions need different output paths.")

    data = load_dataset(args.dataset, split=args.split)

    if args.model is not None:
        model = load_model(args.model)

        if args.family is not None and args.family != model.family:
            parser.error("--family differs from the saved model family.")

        if data.covariate_names != model.covariate_names_:
            raise ValueError("Evaluation requires the fitted covariate names and order.")

        family = model.family

        # Keep covariate effects, the intercept and sigmoid out of the PRS.
        name = getattr(model, "method_name_", "BLocPRS")
        scores = pd.DataFrame({name: model.predict_prs(data)})
    else:
        family = args.family
        scores = load_scores(args.scores, data.sample_ids, args.score_columns)

    if args.reference is not None and args.reference not in scores:
        parser.error("--reference must name one of the evaluated score columns.")

    results = evaluate_scores(data, scores, family, args.split)

    # Keep predictions fixed; resample only the evaluation participants.
    if args.bootstrap:
        uncertainty = bootstrap_metrics(data.y, data.covariates, scores, family=family,
                                        n_boot=args.bootstrap, seed=args.seed)
        summary = uncertainty.method_summary.drop(columns="observed")
        results = results.merge(summary, on="method", how="left", validate="one_to_one")
        results["seed"] = args.seed

        if args.reference is not None:
            paired = paired_comparisons(uncertainty, args.reference)
            paired = paired.drop(columns=["reference", "n_boot"]).rename(columns={
                "bootstrap_se": "difference_se", "ci_lower": "difference_ci_lower",
                "ci_upper": "difference_ci_upper"})
            results = results.merge(paired, on="method", how="left", validate="one_to_one")
            results["reference"] = args.reference

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, sep="\t", index=False, na_rep="NA")

    if args.predictions is not None:
        predictions = pd.DataFrame(data.sample_ids, columns=["FID", "IID"])
        predictions["phenotype"] = data.y
        predictions = pd.concat([predictions, scores.add_prefix("score_")], axis=1)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions, sep="\t", index=False)

    display = ["method", "metric", "value", "n", "status"]
    display += [name for name in ("bootstrap_se", "difference", "difference_se") if name in results]
    print(results[display].to_string(index=False, na_rep="NA"))


if __name__ == "__main__":
    main()
