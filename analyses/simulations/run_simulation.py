"""Fit and evaluate one LDL or asthma simulation on the stored train/test split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from blocprs import BLocPRS
from blocprs.baseline_model import MultiPRS, genomewide_data
from blocprs.metrics import IncrementalR2, NagelkerkeR2

from .generate import prepare_simulation


ROOT = Path(__file__).resolve().parents[2]
PANELS = {"LDL_MA": "LDL", "ASTHMA_MA": "ASTHMA"}


# Select one scenario; the model grid and CV settings match the primary suite.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Stage-5 window-score dataset directory.")
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--architecture", choices=("A0", "A1", "A2"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/simulations/blocprs")
    parser.add_argument("--window-map", type=Path, default=ROOT / "tables/windows_GRCh37_1Mb.tsv")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--block-size", type=int, default=8192)
    return parser.parse_args(argv)


# Isolate the fitted regional predictor using the model's own training scaling.
def regional_prediction(model, data, block_size):
    prediction = np.empty(data.n_samples, dtype=np.float64)
    eta_centered = model.eta_centered_

    for start in range(0, data.n_samples, block_size):
        rows = slice(start, min(start + block_size, data.n_samples))
        scores = model.scaler_.transform_windows(data.window_scores[rows])
        prediction[rows] = np.einsum("nwk,wk->n", scores, eta_centered)

    return prediction


# A constant regional predictor has no estimable Pearson correlation.
def regional_correlation(truth, fitted):
    truth = np.asarray(truth, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    truth = truth - truth.mean()
    fitted = fitted - fitted.mean()
    denominator = np.sqrt(np.dot(truth, truth) * np.dot(fitted, fitted))
    return float(np.dot(truth, fitted) / denominator) if denominator > 0 else np.nan


# Express recovery relative to the known global predictor, without clipping.
def recovery_percentages(values, architecture):
    gap = values["simulation_truth"] - values["true_genomewide_component"]

    if architecture == "A0":
        return np.nan, np.nan, gap, "not_applicable_A0"

    if abs(gap) <= 1e-12:
        return np.nan, np.nan, gap, "zero_truth_gap"

    # Finite-sample noise can put recovery below 0% or above 100%.
    global_value = values["true_genomewide_component"]
    multiprs = 100 * (values["nonwindowed_multiprs"] - global_value) / gap
    blocprs = 100 * (values["blocprs_full"] - global_value) / gap
    return multiprs, blocprs, gap, "ok"


# Keep the generating coefficients, outcomes and frozen scores in one archive.
def save_truth(path, simulation, predictions, regional, args):
    truth = simulation.truth
    arrays = {
        "panel": args.panel, "architecture": args.architecture,
        "replicate": args.replicate, "seed": args.seed, "family": simulation.family,
        "gamma0": truth.gamma0, "gamma": truth.gamma,
        "eta_centered": truth.eta_centered, "delta": truth.delta,
        "active": truth.active, "lambda_sim": truth.lambda_sim,
        "regional_strength_ratio": truth.regional_strength,
        "noise_sd": simulation.noise_sd, "intercept": simulation.intercept,
        "source_sd": simulation.source_sd, "window_mean": simulation.window_mean,
        "source_names": np.asarray(simulation.train.source_names, dtype=str),
        "window_names": np.asarray(simulation.train.window_names, dtype=str),
        "covariate_names": np.asarray(simulation.train.covariate_names, dtype=str),
        "train_y": simulation.train.y, "test_y": simulation.test.y,
        "train_global": simulation.train_global, "test_global": simulation.test_global,
        "train_regional": simulation.train_regional, "test_regional": simulation.test_regional,
        "fitted_test_regional": regional,
        **{f"test_{name}": score for name, score in predictions.items()},
    }

    for split in ("train", "test"):
        ids = getattr(simulation, split).sample_ids
        if ids is not None:
            arrays[f"{split}_ids"] = np.asarray(ids, dtype=str)

    np.savez_compressed(path, **arrays)


# Fit both estimators on training rows and evaluate frozen scores on test rows.
def run_simulation(args):
    output = args.output_dir / args.panel / args.architecture / f"replicate_{args.replicate:02d}"

    # Refuse an existing scenario directory, including an incomplete earlier run.
    output.mkdir(parents=True, exist_ok=False)
    simulation = prepare_simulation(args.dataset, args.window_map, args.panel,
                                    args.architecture, args.replicate, args.seed, args.block_size)
    train_global = genomewide_data(simulation.train, args.block_size)
    test_global = genomewide_data(simulation.test, args.block_size)

    multiprs = MultiPRS(simulation.family, variant="unregularized", block_size=args.block_size)
    multiprs.fit(train_global)

    model = BLocPRS(simulation.family, lambda_grid=np.logspace(-3, 3, 25), cv_folds=5,
                   cv_max_samples=50000, random_state=42, block_size=args.block_size)
    model.fit(simulation.train)

    # Each comparison uses the same held-out participants and covariate-only fit.
    metric_type = IncrementalR2 if simulation.family == "gaussian" else NagelkerkeR2
    metric = metric_type(simulation.test.y, simulation.test.covariates)
    single_values = [metric.score(test_global.window_scores[:, 0, k])
                     for k in range(test_global.n_sources)]

    # The single-source reference is explicitly an oracle selected on test outcomes.
    best_source = int(np.argmax(single_values))
    predictions = {
        "oracle_best_single": test_global.window_scores[:, 0, best_source],
        "nonwindowed_multiprs": multiprs.predict_prs(test_global),
        "blocprs_full": model.predict_prs(simulation.test),
        "simulation_truth": simulation.test_global + simulation.test_regional,
        "true_genomewide_component": simulation.test_global,
    }
    values = {"oracle_best_single": float(single_values[best_source])}
    values.update({name: metric.score(score) for name, score in predictions.items()
                   if name != "oracle_best_single"})
    regional = regional_prediction(model, simulation.test, args.block_size)
    correlation = regional_correlation(simulation.test_regional, regional)
    multiprs_recovery, blocprs_recovery, gap, status = recovery_percentages(values, args.architecture)

    result = {
        "panel": PANELS[args.panel], "base_panel": args.panel,
        "architecture": args.architecture, "replicate": args.replicate, "seed": args.seed,
        "family": simulation.family,
        "Metric": "Incremental_R2" if simulation.family == "gaussian" else "Nagelkerke_R2",
        "regional_strength_ratio": simulation.truth.regional_strength,
        "n_train": simulation.train.n_samples, "n_test": simulation.test.n_samples,
        "selected_lambda": model.selected_lambda_,
        "cv_folds": 5, "cv_max_samples": 50000, "cv_seed": 42,
        "oracle_best_source": test_global.source_names[best_source],
        **values,
        "nonwindowed_multiprs_recovery_pct": multiprs_recovery,
        "blocprs_full_recovery_pct": blocprs_recovery,
        "true_regional_gain": gap, "recovery_status": status,
        "heldout_regional_predictor_correlation": correlation,
    }

    model.save(output / "model.npz")
    multiprs.save(output / "multiprs.npz")
    save_truth(output / "truth.npz", simulation, predictions, regional, args)

    # Write the result row last, only after fitting, evaluation and archive saving succeed.
    pd.DataFrame([result]).to_csv(output / "results.tsv", sep="\t", index=False, na_rep="NA")
    print(f"{args.panel} {args.architecture} replicate={args.replicate:02d} "
          f"lambda={model.selected_lambda_:g} MultiPRS={values['nonwindowed_multiprs']:.6g} "
          f"BLocPRS={values['blocprs_full']:.6g} results={output / 'results.tsv'}")
    return result


if __name__ == "__main__":
    run_simulation(parse_args())
