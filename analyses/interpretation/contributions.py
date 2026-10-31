"""Summarize fitted BLocPRS contributions using training-score moments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from blocprs import BLocPRS, load_dataset, regional_coefficients


REFERENCE_EDGES = [0, 50, 70, 85, 90, 95, 99, 100]
REFERENCE_BINS = ["0-50", "50-70", "70-85", "85-90", "90-95", "95-99", "99-100"]


# Accumulate centered population moments without a participant-by-window cache.
class _Moments:
    def __init__(self):
        self.n = 0

    # Merge block means, sums of squares and optional covariance with a total score.
    def add(self, values, total=None):
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        ss = np.sum(centered * centered, axis=0)
        if total is not None:
            total = np.asarray(total, dtype=np.float64)
            total_mean = float(total.mean())
            total_centered = total - total_mean
            total_ss = float(total_centered @ total_centered)
            cross = centered.T @ total_centered

        if not self.n:
            self.mean, self.ss = mean, ss
            if total is not None:
                self.total_mean, self.total_ss, self.cross = total_mean, total_ss, cross
        else:
            combined = self.n + n
            weight = self.n * n / combined
            difference = mean - self.mean

            # Correct for different block means when combining sums of squares.
            self.ss += ss + difference * difference * weight
            self.mean += difference * (n / combined)
            if total is not None:
                total_difference = total_mean - self.total_mean
                self.cross += cross + difference * total_difference * weight
                self.total_ss += total_ss + total_difference * total_difference * weight
                self.total_mean += total_difference * (n / combined)
        self.n += n

    # Population variance matches the model's training scaling convention.
    @property
    def variance(self):
        return self.ss / self.n

    # Signed covariance retains both within-window and cross-window dependence.
    @property
    def covariance(self):
        return self.cross / self.n

    # The denominator is variance of the total genetic component, not the phenotype.
    @property
    def total_variance(self):
        return self.total_ss / self.n


# Produce centered-raw reference and regional components in bounded participant blocks.
def iter_components(model, data, block_size=1024):
    if (data.source_names != model.source_names_ or data.window_names != model.window_names_
        or block_size < 1):
        raise ValueError("Dataset sources/windows must match the model, with a positive block size.")

    gamma, eta_centered = model.gamma_, model.eta_centered_
    for start in range(0, data.n_samples, block_size):
        rows = slice(start, min(start + block_size, data.n_samples))
        scores = model.scaler_.transform_windows(data.window_scores[rows])

        # Sum over sources, retaining one genetic contribution per participant and window.
        reference = np.einsum("iwk,k->iw", scores, gamma, dtype=np.float64)
        adjustment = np.einsum("iwk,wk->iw", scores, eta_centered, dtype=np.float64)
        yield rows, scores, reference, adjustment


# Match canonical global window IDs to the model's ordered feature names.
def window_coordinates(window_map, names):
    table = pd.read_csv(window_map, sep="\t") if not isinstance(window_map, pd.DataFrame) else window_map.copy()
    if "window" not in table:
        table["window"] = [f"w{int(value):04d}" for value in table.window_id_global]
    table["window"] = table.window.astype(str)
    table["CHR"] = table.CHR.astype(str).str.removeprefix("chr").astype(int)
    table = table.set_index("window", verify_integrity=True).loc[list(names), ["CHR", "start_bp", "end_bp"]]

    if (table.start_bp.lt(1).any() or table.end_bp.lt(table.start_bp).any()
        or table[["start_bp", "end_bp"]].isna().any().any()):
        raise ValueError("Window coordinates must be one-based inclusive intervals.")

    for _, group in table.groupby("CHR"):
        ordered = group.sort_values("start_bp")
        if np.any(ordered.start_bp.to_numpy()[1:] <= ordered.end_bp.to_numpy()[:-1]):
            raise ValueError("Genomic windows overlap.")
    return table.reset_index()


# Stable ranks retain model window order when contribution magnitudes tie.
def descending_rank(values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Window-ranking values must be finite.")

    ranks = np.empty(len(values), dtype=int)
    ranks[np.argsort(-values, kind="stable")] = np.arange(1, len(values) + 1)
    return ranks


# Confirm saved training contributions belong to this exact coefficient layout.
def validate_context(model, context):
    expected = regional_coefficients(model).rename(columns={"source": "base"})
    keys = ["window", "base"]
    observed = context.copy()
    observed["window"] = observed.window.astype(str)
    observed["base"] = observed.base.astype(str)
    index = pd.MultiIndex.from_frame(expected[keys])
    observed = observed.set_index(keys, verify_integrity=True)

    if set(observed.index) != set(index):
        raise ValueError("Contribution windows and sources differ from the model.")

    observed = observed.loc[index].reset_index()
    columns = ["gamma", "eta_centered", "c_eff", "raw_multiplier"]
    if (not np.allclose(observed[columns], expected[columns], rtol=1e-10, atol=1e-12)
        or not observed.n_training.eq(model.n_training_samples_).all()
        or not np.isfinite(observed.model_prs_sd).all() or observed.model_prs_sd.lt(0).any()):
        raise ValueError("Training contributions do not match the fitted model.")

    return observed


# Compute training moments and both reference- and adjustment-based window rankings.
def compute_contributions(model, data, window_map, phenotype, block_size=1024):
    if data.n_samples != model.n_training_samples_:
        raise ValueError("Use the full dataset training split used to fit this model.")

    source_moments = _Moments()
    stats = {name: _Moments() for name in ("R", "D", "P", "D_to_P")}
    raw_sum = np.zeros((data.n_windows, data.n_sources), dtype=float)
    genomewide = _Moments()
    for rows, scores, reference, adjustment in iter_components(model, data, block_size):
        raw = np.asarray(data.window_scores[rows], dtype=np.float32)
        raw_sum += np.sum(raw, axis=0, dtype=np.float64)
        genomewide.add(np.sum(raw, axis=1, dtype=np.float64))
        source_moments.add(scores.reshape(len(scores), -1))
        effective = reference + adjustment
        totals = [values.sum(axis=1) for values in (reference, adjustment, effective)]
        for name, values, total in zip(("R", "D", "P", "D_to_P"),
                                       (reference, adjustment, effective, adjustment), (*totals, totals[2])):
            stats[name].add(values, total)

    # Detect a different training cohort before its moments become selection ranks.
    raw_mean = raw_sum / data.n_samples
    source_sd = np.sqrt(genomewide.variance).astype(np.float32)
    source_sd[source_sd == 0] = 1
    if (not np.allclose(raw_mean, model.scaler_.window_mean, rtol=1e-6, atol=1e-8)
        or not np.allclose(source_sd, model.scaler_.source_sd, rtol=1e-5, atol=1e-8)):
        raise ValueError("Dataset training moments differ from the saved model scaling.")

    coordinates = window_coordinates(window_map, model.window_names_)
    widths = coordinates.end_bp - coordinates.start_bp + 1
    width = int(widths.iloc[0])
    window_size = f"mb{width // 1000000}" if widths.eq(width).all() and width % 1000000 == 0 else "custom"
    context = regional_coefficients(model).rename(columns={"source": "base"})
    context = context.merge(coordinates, on="window", validate="many_to_one", sort=False)
    context["model_prs_var"] = source_moments.variance
    context["model_prs_sd"] = np.sqrt(context.model_prs_var)
    context["phenotype"], context["family"] = phenotype, "binary" if model.family == "binomial" else "continuous"
    context["expert_label"] = context.base
    context["n_training"] = data.n_samples
    context["window_size"] = window_size
    for name, coefficient in (("reference", "gamma"), ("adjustment", "eta_centered"), ("effective", "c_eff")):
        context[name] = context[coefficient] * context.model_prs_sd

    # Absolute contributions define ranking magnitude, not variance explained.
    masses = context.assign(G=context.reference.abs(), A=context.adjustment.abs(), E=context.effective.abs())
    masses = masses.groupby("window", sort=False)[["G", "A", "E"]].sum().reindex(model.window_names_)
    windows = coordinates.merge(masses.reset_index(), on="window", validate="one_to_one", sort=False)
    windows["phenotype"], windows["family"] = phenotype, context.family.iloc[0]
    windows["n_training"] = data.n_samples
    windows["window_size"] = window_size
    windows["reference_rank"] = descending_rank(windows.G)
    windows["adjustment_rank"] = descending_rank(windows.A)
    windows["reference_percentile"] = windows.G.rank(method="average", pct=True) * 100
    windows["bin"] = pd.cut(windows.reference_percentile, REFERENCE_EDGES, labels=REFERENCE_BINS, include_lowest=True)
    floor = max(1e-30, 1e-12 * max(stats["R"].total_variance, stats["P"].total_variance))
    sd = context.model_prs_sd.to_numpy().reshape(data.n_windows, data.n_sources)
    for name, coefficient in (("R", model.gamma_[None, :]), ("D", model.eta_centered_), ("P", model.c_eff_)):
        moments = stats[name]
        diagonal = np.sum((coefficient * sd) ** 2, axis=1)
        windows[f"SD_{name}"] = np.sqrt(moments.variance)
        windows[f"Cov_{name}"] = moments.covariance

        # Cov(window, total) / Var(total) is signed and includes cross-window dependence.
        windows[f"q_{name}"] = moments.covariance / moments.total_variance if moments.total_variance > floor else np.nan
        windows[f"V_diag_{name}"], windows[f"V_net_{name}"] = diagonal, moments.variance

        # The difference retains covariance between source contributions within this window.
        windows[f"V_cross_{name}"] = moments.variance - diagonal
        windows[f"net_diag_ratio_{name}"] = np.divide(moments.variance, diagonal,
                                                     out=np.full_like(diagonal, np.nan), where=diagonal > 0)
    windows["q_D_to_P"] = (stats["D_to_P"].covariance / stats["P"].total_variance
                            if stats["P"].total_variance > floor else np.nan)
    return context, windows


# Summarize signed profiles and magnitude ratios using reference-percentile bins.
def summarize_contributions(context):
    frame = context.copy()
    keys = ["phenotype", "window"]

    if frame.duplicated(keys + ["base"]).any():
        raise ValueError("Duplicate phenotype/window/source contributions.")

    for name, coefficient in (("reference", "gamma"), ("adjustment", "eta_centered"), ("effective", "c_eff")):
        frame[name] = frame[coefficient] * frame.model_prs_sd
    strengths = frame.assign(G=frame.reference.abs(), A=frame.adjustment.abs(), E=frame.effective.abs())
    strengths = strengths.groupby(keys, sort=False)[["G", "A", "E"]].sum()
    strengths["percentile"] = strengths.groupby("phenotype").G.rank(method="average", pct=True) * 100
    strengths["bin"] = pd.cut(strengths.percentile, REFERENCE_EDGES, labels=REFERENCE_BINS, include_lowest=True)
    totals = strengths.groupby(["phenotype", "bin"], observed=False, sort=False)[["G", "A"]].sum().reset_index()
    totals["ratio"] = np.divide(100 * totals.A, totals.G, out=np.full(len(totals), np.nan), where=totals.G > 0)
    frame = frame.drop(columns=[name for name in strengths.columns if name in frame])
    frame = frame.merge(strengths.reset_index(), on=keys, validate="many_to_one", sort=False)
    means = frame.groupby(["phenotype", "bin", "expert_label"], observed=True)[
        ["reference", "adjustment", "effective"]].mean().reset_index()

    # Zero-mass windows have undefined source shares, not zero share changes.
    top = frame.loc[frame.bin.eq("99-100")].copy()

    if top.G.le(0).any() or top.E.le(0).any():
        raise ValueError("Top-window source shares require positive reference and effective magnitudes.")

    top["reference_share"] = top.reference.abs() / top.G
    top["effective_share"] = top.effective.abs() / top.E
    top["shift"] = 100 * (top.effective_share - top.reference_share)
    top["sign_reversal"] = top.reference * top.effective < 0
    return means, totals[["phenotype", "bin", "ratio"]], top


# Compare reference-magnitude, net-SD and signed-covariance rankings on training data.
def covariance_bins(windows):
    records = []
    for phenotype, group in windows.groupby("phenotype", sort=False):
        for ranking, column in (("original", "G"), ("net_sd", "SD_R"), ("signed_covariance", "Cov_R")):
            if ranking == "signed_covariance" and group.q_R.isna().any():
                continue

            percentile = group[column].rank(method="average", pct=True) * 100
            bins = pd.cut(percentile, REFERENCE_EDGES, labels=REFERENCE_BINS, include_lowest=True)
            for label in REFERENCE_BINS:
                selected = group.loc[bins == label]
                reference, adjustment = selected.G.sum(), selected.A.sum()
                net_reference, net_adjustment = selected.SD_R.sum(), selected.SD_D.sum()
                records.append({"phenotype": phenotype, "ranking": ranking, "bin": label,
                                "n_windows": len(selected),
                                "original_ratio": 100 * adjustment / reference if reference > 0 else np.nan,
                                "net_sd_ratio": 100 * net_adjustment / net_reference if net_reference > 0 else np.nan})
    return pd.DataFrame(records)


# Save two reusable tables; all rankings and moments come from the stored training split.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phenotype", required=True, help="Analysis ID, e.g. LDL_MA; independent of the dataset directory name.")
    parser.add_argument("--window-map", type=Path, default=Path("tables/windows_GRCh37_1Mb.tsv"))
    parser.add_argument("--output-dir", type=Path, help="Default: data/analyses/interpretation/<phenotype>.")
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()
    model = BLocPRS.load(args.model)
    data = load_dataset(args.dataset, split="train")
    context, windows = compute_contributions(model, data, args.window_map, args.phenotype, args.block_size)
    output = args.output_dir or Path("data/analyses/interpretation") / args.phenotype
    output.mkdir(parents=True, exist_ok=True)
    context.to_csv(output / "contributions.tsv", sep="\t", index=False)
    windows.to_csv(output / "windows.tsv", sep="\t", index=False, na_rep="NA")
    print(f"{args.phenotype}: {data.n_samples:,} training participants, "
          f"{data.n_windows:,} windows and {data.n_sources} sources -> {output}")


if __name__ == "__main__":
    main()
