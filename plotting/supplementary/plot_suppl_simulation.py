"""Prediction and locus-specific recovery across three simulation architectures."""

import argparse
from math import ceil, floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..plot_utils import save_figure, style, style_axis


architectures = ("A0", "A1", "A2")
strengths = {"A0": 0, "A1": .02, "A2": .10}
panels = {
    "LDL": ("ldl", "LDL_MA", "Incremental_R2", "Continuous (LDL-C)", r"Incremental $R^2$"),
    "ASTHMA": ("asthma", "ASTHMA_MA", "Nagelkerke_R2", "Binary (asthma)", r"Nagelkerke $R^2$"),
}
methods = ("oracle_best_single", "nonwindowed_multiprs", "blocprs_full", "simulation_truth")
method_labels = ("Oracle best single-source PRS", "MultiPRS", "BLocPRS", "Simulation truth")
colors = ("#A6A6A6", "#6BAED6", "#6A51A3", "#D6A84F")
correlation_column = "heldout_regional_predictor_correlation"


# Read the combined result table from the packaged simulation suite.
def load_combined_results(path):
    from analyses.simulations.summarize import result_columns, validate_results

    data = pd.read_csv(path, sep="\t", float_precision="round_trip")[result_columns]
    replicates = data.replicate.nunique()

    if replicates < 2:
        raise ValueError("The simulation figure requires at least two replicates per architecture")

    validate_results(data, [spec[1] for spec in panels.values()], replicates)
    return data.sort_values(["panel", "architecture", "replicate"])


# Display the replicate mean, its SEM, and every replicate at its observed value.
def mean_bar(ax, center, values, width, color, truth=False):
    values = np.asarray(values, dtype=float)

    # A failed replicate makes the summary NE; it is not dropped from the average.
    if not np.isfinite(values).all():
        ax.annotate("NE", (center, 0), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.7, color="#666666")
        return 0

    mean = values.mean()

    # This is the SE of the replicate mean, not participant-level prediction uncertainty.
    sem = values.std(ddof=1) / np.sqrt(len(values))
    ax.bar(center, mean, width=width, color=to_rgba(color, .20 if truth else .90),
           edgecolor=color if truth else "#666666", linewidth=.55 if truth else .20, zorder=2)
    ax.errorbar(center, mean, yerr=sem, fmt="none", ecolor="#444444", elinewidth=.48, capsize=0, zorder=4)
    ax.scatter(center + np.linspace(-width*.22, width*.22, len(values)), values, s=2,
               facecolor="#444444", edgecolor="none", alpha=.35, zorder=3)
    return max(values.max(), mean + sem)


# Keep the recovery and correlation scales common to continuous and binary panels.
def build_figure(data):
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 112 / 25.4))
    fig.subplots_adjust(left=.09, right=.985, bottom=.13, top=.815, wspace=.38, hspace=.32)
    recovery = data[[method + "_recovery_pct" for method in methods[1:3]]].to_numpy().ravel()
    recovery = recovery[np.isfinite(recovery)]
    recovery_limits = ((10*floor(min(0, recovery.min()-3)/10), 10*ceil(max(105, recovery.max()+5)/10))
                       if len(recovery) else (-10, 110))
    correlation = data.loc[data.architecture.ne("A0"), correlation_column].dropna()
    correlation_min = min(0, .1*floor((correlation.min()-.03)/.1)) if len(correlation) else 0
    for row, (panel, spec) in enumerate(panels.items()):
        selected = data.loc[data.panel.eq(panel)]
        for ax in axes[row]:
            style_axis(ax)
            ax.set_xlim(-.5, 2.5)
            ax.set_xticks(range(3), architectures)
            ax.tick_params(labelsize=5.8, length=2)
        maximum = 0
        for index, (method, color) in enumerate(zip(methods, colors)):
            for x, architecture in enumerate(architectures):
                values = selected.loc[selected.architecture.eq(architecture), method]
                maximum = max(maximum, mean_bar(axes[row, 0], x+(index-1.5)*.17, values, .15,
                                                color, truth=method == "simulation_truth"))
        axes[row, 0].set_ylim(0, .02*ceil(maximum*1.04/.02))
        axes[row, 0].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=(1, 2, 5, 10)))
        axes[row, 0].set_ylabel(spec[4], fontsize=6.4, labelpad=4)
        for index, (method, color) in enumerate(zip(methods[1:3], colors[1:3])):
            for x, architecture in enumerate(architectures[1:], start=1):
                values = selected.loc[selected.architecture.eq(architecture), method + "_recovery_pct"]
                mean_bar(axes[row, 1], x+(-.17, .17)[index], values, .28, color)
        ax = axes[row, 1]
        ax.set_ylim(recovery_limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylabel("Locus-specific adjustment\ngain recovery (%)", fontsize=6.4, labelpad=4)
        ax.axhline(0, color="#888888", linewidth=.55, zorder=1)
        ax.axhline(100, color="#8A7138", linewidth=.55, linestyle=(0, (2, 2)), zorder=1)
        ax = axes[row, 2]
        for x, architecture in enumerate(architectures[1:], start=1):
            values = selected.loc[selected.architecture.eq(architecture), correlation_column]
            mean_bar(ax, x, values, .28, colors[2])
        ax.set_ylim(correlation_min, 1.06)
        ax.set_yticks(np.arange(0, 1.001, .25) if correlation_min == 0 else np.linspace(correlation_min, 1, 5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_ylabel("Correlation with truth", fontsize=6.4, labelpad=4)
        ax.axhline(1, color="#8A7138", linewidth=.55, linestyle=(0, (2, 2)), zorder=1)
        for ax in axes[row, 1:]:
            ax.annotate("N/A", (0, 0), xytext=(0, 5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=5.7, color="#666666")
        box = axes[row, 0].get_position()
        fig.text(.018, (box.y0+box.y1)/2, spec[3], rotation=90, ha="center", va="center", fontsize=6.8)
    titles = ("Prediction performance", "Locus-specific adjustment\ngain recovery",
              "Locus-specific adjustment\npredictor recovery")
    for column, title in enumerate(titles):
        box = axes[0, column].get_position()
        fig.text(box.x0, .86, "abc"[column], fontsize=8, fontweight="bold", va="bottom")
        fig.text((box.x0+box.x1)/2+.008, .86, title, fontsize=7, ha="center", va="bottom")
    handles = [Patch(facecolor=to_rgba(color, .20) if index == 3 else color,
                     edgecolor=color if index == 3 else "none", linewidth=.55, label=label)
               for index, (label, color) in enumerate(zip(method_labels, colors))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .975), ncol=4, frameon=False,
               fontsize=6, columnspacing=1.6, handlelength=1.7, handletextpad=.45)
    fig.text(.5, .035, "Simulation architecture", fontsize=6.4, ha="center")
    return fig


# Export the simulation comparison without fitting or resampling.
def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "figures/supplementary")
    parser.add_argument("--results", type=Path, help="Combined results.tsv from analyses.simulations.summarize.")
    args = parser.parse_args()
    data = load_combined_results(args.results or args.data_root / "simulations/blocprs/results.tsv")
    with plt.rc_context(style):
        save_figure(build_figure(data), args.output_dir / "suppl_simulation.pdf")


if __name__ == "__main__":
    main()
