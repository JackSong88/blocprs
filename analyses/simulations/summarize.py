"""Summarize every planned replicate of the three BLocPRS simulation endpoints."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


panels = {"LDL_MA": ("LDL", "Incremental_R2", "gaussian"),
          "ASTHMA_MA": ("ASTHMA", "Nagelkerke_R2", "binomial")}
strengths = {"A0": 0.0, "A1": 0.02, "A2": 0.10}
prediction_columns = ("oracle_best_single", "nonwindowed_multiprs", "blocprs_full", "simulation_truth")
correlation_column = "heldout_regional_predictor_correlation"
result_columns = [
    "panel", "base_panel", "architecture", "replicate", "seed", "family", "Metric",
    "regional_strength_ratio", "n_train", "n_test", "selected_lambda",
    "cv_folds", "cv_max_samples", "cv_seed", "oracle_best_source",
    *prediction_columns, "true_genomewide_component", "nonwindowed_multiprs_recovery_pct",
    "blocprs_full_recovery_pct", correlation_column, "recovery_status",
]


# Check scenario identity and the common-reference recovery calculation.
def validate_results(data, selected_panels, replicates):
    expected = {(panel, architecture, replicate) for panel in selected_panels
                for architecture in strengths for replicate in range(replicates)}
    observed = set(zip(data.base_panel, data.architecture, data.replicate))

    if len(data) != len(expected) or observed != expected:
        raise ValueError("Expected one result for every requested panel, architecture, and replicate")

    for base_panel, group in data.groupby("base_panel", sort=False):
        panel, metric, family = panels[base_panel]

        if not group.panel.eq(panel).all() or not group.Metric.eq(metric).all() or not group.family.eq(family).all():
            raise ValueError(f"Inconsistent panel, model family or evaluation metric: {base_panel}")
        if group[["seed", "n_train", "n_test"]].nunique(dropna=False).ne(1).any():
            raise ValueError(f"Mixed seeds or sample sizes: {base_panel}")

    cv_settings = pd.Series({"cv_folds": 5, "cv_max_samples": 50000, "cv_seed": 42})

    if not data[cv_settings.index].eq(cv_settings).all().all():
        raise ValueError("Simulation results use different CV settings from the primary suite")
    if not np.allclose(data.regional_strength_ratio, data.architecture.map(strengths), rtol=1e-6, atol=1e-10):
        raise ValueError("Simulation architectures have unexpected regional-effect strengths")
    if not np.isfinite(data[[*prediction_columns, "true_genomewide_component"]]).all().all():
        raise ValueError("A simulation has a nonfinite prediction metric")

    gap = data.simulation_truth - data.true_genomewide_component
    null = data.architecture.eq("A0")
    usable = ~null & gap.abs().gt(1e-12)
    status = np.where(null, "not_applicable_A0", np.where(usable, "ok", "zero_truth_gap"))

    if not np.allclose(gap.loc[null], 0, rtol=0, atol=1e-12) or not data.recovery_status.eq(status).all():
        raise ValueError("Inconsistent true regional gap or recovery status")

    for method in ("nonwindowed_multiprs", "blocprs_full"):
        recovery = data[method + "_recovery_pct"]
        reconstructed = recovery.loc[usable] * gap.loc[usable]
        numerator = 100 * (data.loc[usable, method] - data.loc[usable, "true_genomewide_component"])

        if not recovery.loc[~usable].isna().all() or not np.isfinite(recovery.loc[usable]).all():
            raise ValueError(f"Inconsistent nonestimable recovery: {method}")
        if not np.allclose(reconstructed, numerator, rtol=1e-8, atol=1e-10):
            raise ValueError(f"Recovery does not use the common true genome-wide reference: {method}")

    correlation = data[correlation_column]

    if not correlation.loc[null].isna().all() or not (correlation.isna() | correlation.between(-1, 1)).all():
        raise ValueError("Invalid regional-predictor correlation")


# Read explicit expected paths; a missing or failed replicate stops aggregation.
def load_results(results_dir, selected_panels, replicates):
    frames = []
    for base_panel in selected_panels:
        for architecture in strengths:
            for replicate in range(replicates):
                path = results_dir / base_panel / architecture / f"replicate_{replicate:02d}" / "results.tsv"
                frame = pd.read_csv(path, sep="\t", float_precision="round_trip")[result_columns]

                if len(frame) != 1 or tuple(frame.loc[0, ["base_panel", "architecture", "replicate"]]) != (base_panel, architecture, replicate):
                    raise ValueError(f"Result identity does not match its directory: {path}")

                frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    validate_results(data, selected_panels, replicates)
    return data


# Keep nonestimable endpoints visible instead of averaging a successful subset.
def summarize_results(data):
    endpoints = [("prediction", method, method) for method in prediction_columns]
    endpoints += [("regional_gain_recovery_pct", method, method + "_recovery_pct")
                  for method in ("nonwindowed_multiprs", "blocprs_full")]
    endpoints += [("regional_predictor_correlation", "blocprs_full", correlation_column)]
    rows = []
    for (panel, base_panel, architecture, metric), group in data.groupby(
            ["panel", "base_panel", "architecture", "Metric"], sort=False):
        for endpoint, method, column in endpoints:
            values = group[column].to_numpy(dtype=float)
            n = int(np.isfinite(values).sum())
            complete = n == len(values)

            # SEM is across simulation replicates, not participant bootstrap uncertainty.
            rows.append({
                "panel": panel, "base_panel": base_panel, "architecture": architecture,
                "Metric": metric, "endpoint": endpoint, "method": method,
                "mean": float(values.mean()) if complete else np.nan,
                "sem": float(values.std(ddof=1) / np.sqrt(n)) if complete and n > 1 else np.nan,
                "n": n, "n_total": len(values),
            })
    return pd.DataFrame(rows)


# Write one plot-ready table and one endpoint summary.
def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "data/simulations/blocprs")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--panels", nargs="+", choices=panels, default=list(panels))
    args = parser.parse_args()

    if args.replicates < 1 or len(args.panels) != len(set(args.panels)):
        parser.error("Choose at least one replicate and do not repeat panels")

    data = load_results(args.results_dir, args.panels, args.replicates)
    data.to_csv(args.results_dir / "results.tsv", sep="\t", index=False)
    summarize_results(data).to_csv(args.results_dir / "summary.tsv", sep="\t", index=False)
    print(f"Summarized {len(data)} simulations: {args.results_dir}")


if __name__ == "__main__":
    main()
