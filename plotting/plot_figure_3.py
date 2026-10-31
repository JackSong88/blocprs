"""Plot training-size growth and paired BLocPRS advantage from raw evaluations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import MaxNLocator

from .plot_utils import family_colors, save_figure, style, style_axis
from .utils import load_training_sizes, parse_args


sizes = [2500, 5000, 12500, 25000, 50000, 100000, "full"]
size_labels = ["2.5k", "5k", "12.5k", "25k", "50k", "100k", "Full\n(~330k)"]
models = ["Nonwindowed MultiPRS", "BLocPRS mb1"]
cohorts = ["UKBB held-out", "Full CaG transfer"]
families = ["continuous", "binary"]
phenotypes = {
    "continuous": [
        "LDL_MA", "DBP_MA", "HEIGHT_MA", "LOG_BMI_MA", "URT_SEX", "WHR_SEX",
        "LOG_CRTN_SEX", "LOG_HDL_MA", "TC_MA", "FEV1_FVC_MA", "SBP_MA",
        "LOG_TG_MA", "LOG_TST_SEX",
    ],
    "binary": [
        "ASTHMA_MA", "CAD_MT", "AF_MT", "ASTHMA_MT", "GOUT_MT", "HF_MT",
        "HTN_MT", "STR_MT", "T1D_MT", "T2D_MA", "T2D_MT", "DEM_MT",
    ],
}




# Eligibility uses full-training MultiPRS, not the smaller-sample results.
def load_inputs(path):
    data = load_training_sizes(path)
    full = data.loc[data["size"].eq("full")]
    paired = full.pivot(index=["cohort", "phenotype"], columns="model", values="value")
    eligible = paired.index[paired["Nonwindowed MultiPRS"].ge(.01) & paired["BLocPRS mb1"].notna()]
    data = data.loc[pd.MultiIndex.from_frame(data[["cohort", "phenotype"]]).isin(eligible)].copy()
    data = data.loc[~(data.cohort.eq("cartagene") & data.family.eq("binary") & data.cases.lt(100))]
    data["Cohort"] = data.cohort.map(dict(zip(["ukbb", "cartagene"], cohorts)))
    return data.rename(columns={"family": "Family", "phenotype": "Phenotype", "size": "N train",
                                "model": "Model", "value": "Value"})


def relative_change(value, reference):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if not np.isfinite(value).all() or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("Relative changes require finite values and positive denominators")
    return 100.0 * (value / reference - 1.0)


def summarize_inputs(data):

    # Each model's growth is relative to its own N = 2,500 fit, before aggregation.
    keys = ["Cohort", "Family", "Phenotype", "Model"]
    baseline = data.loc[data["N train"].eq(2500), keys + ["Value"]].rename(
        columns={"Value": "Baseline R2"})
    points = data.merge(baseline, on=keys, validate="many_to_one")
    points["Relative change (%)"] = relative_change(points.Value, points["Baseline R2"])
    growth = points.groupby(["Cohort", "Family", "Model", "N train"], sort=False).agg(
        Median=("Relative change (%)", "median"),
        Mean=("Relative change (%)", "mean"),
        N=("Phenotype", "nunique"),
    ).reset_index()

    # Pair phenotype-panel analyses at each N before taking relative-advantage medians.
    keys = ["Cohort", "Family", "Phenotype", "N train"]
    blocprs = points.loc[points.Model.eq("BLocPRS mb1"), keys + ["Value"]]
    multiprs = points.loc[points.Model.eq("Nonwindowed MultiPRS"), keys + ["Value"]]
    paired = blocprs.merge(multiprs, on=keys, validate="one_to_one",
                           suffixes=("_BLocPRS", "_MultiPRS"))
    paired["Relative advantage (%)"] = relative_change(
        paired.Value_BLocPRS, paired.Value_MultiPRS)
    difference = paired.Value_BLocPRS - paired.Value_MultiPRS

    # Wins and ties use the saved point estimates, not significance tests or rounded labels.
    paired["Win"] = difference.gt(0)
    paired["Loss"] = difference.lt(0)
    paired["Tie"] = difference.eq(0)
    keys = ["Cohort", "Family", "N train"]
    advantage = paired.groupby(keys, sort=False).agg(
        Median=("Relative advantage (%)", "median"),
        Mean=("Relative advantage (%)", "mean"),
        N=("Phenotype", "count"),
    ).reset_index()
    wins = paired.groupby(keys, sort=False).agg(
        Wins=("Win", "sum"), Losses=("Loss", "sum"), Ties=("Tie", "sum"),
        Total=("Phenotype", "count"),
    ).reset_index()
    wins["Fraction favoring BLocPRS"] = wins.Wins / wins.Total
    return points, growth, paired, advantage, wins


def add_win_fraction(axis, x, y, fraction, color):

    # Marker position is median advantage; filled area is wins/total, including ties.
    drawing = DrawingArea(8, 8, 0, 0)
    drawing.add_artist(Circle((4, 4), 2.75, facecolor="#F3F3F3", edgecolor="none"))
    if fraction > 0:
        drawing.add_artist(Wedge((4, 4), 2.75, 90 - 360 * fraction, 90,
                                facecolor=color, edgecolor="none"))
    drawing.add_artist(Circle((4, 4), 2.75, facecolor="none", edgecolor="#777777", linewidth=0.3))
    axis.add_artist(AnnotationBbox(drawing, (x, y), frameon=False, xycoords="data",
                                  box_alignment=(0.5, 0.5), pad=0, zorder=6))


def plot_limits(values):
    values = np.asarray(values, dtype=float)
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    padding = max(1.0, 0.075 * (high - low))
    return low - padding, high + padding


def build_figure(growth, paired, advantage, wins):
    fig = plt.figure(figsize=(183 / 25.4, 122 / 25.4))
    grid = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.840, bottom=0.090,
                           hspace=0.40, wspace=0.19)
    bounds = grid[:, :].get_position(fig)
    center_x = (bounds.x0 + bounds.x1) / 2
    x = np.arange(len(sizes))
    offsets = {"continuous": -0.20, "binary": 0.20}

    # Match each column across cohorts, including every displayed boxplot outlier.
    growth_limits = plot_limits(growth.Median)
    advantage_limits = plot_limits(paired["Relative advantage (%)"])
    headers = ["Held-out UK Biobank", r"UK Biobank $\rightarrow$ CARTaGENE transfer"]
    for row, cohort in enumerate(cohorts):
        left, right = [fig.add_subplot(grid[row, column]) for column in range(2)]
        for family in families:
            color = family_colors[family]
            for model in models:
                baseline = model == "Nonwindowed MultiPRS"
                values = growth.loc[
                    growth.Cohort.eq(cohort) & growth.Family.eq(family) & growth.Model.eq(model)
                ].set_index("N train").reindex(sizes)
                left.plot(x, values.Median, color=color,
                          linestyle="--" if baseline else "-", linewidth=0.9 if baseline else 1.25,
                          marker="o" if baseline else "s", markersize=2.8,
                          markerfacecolor="white" if baseline else color,
                          markeredgewidth=0.55, zorder=4)
            cell = paired.loc[paired.Cohort.eq(cohort) & paired.Family.eq(family)]

            # These boxes summarize variation across phenotypes, not uncertainty within a phenotype.
            statistics = [boxplot_stats(
                cell.loc[cell["N train"].eq(size), "Relative advantage (%)"].to_numpy(),
                whis=1.5, autorange=False)[0] for size in sizes]
            positions = x + offsets[family]
            right.bxp(statistics, positions=positions, widths=0.28, manage_ticks=False,
                      patch_artist=True, showfliers=True,
                      boxprops={"facecolor": to_rgba(color, 0.09),
                                "edgecolor": to_rgba(color, 0.40), "linewidth": 0.4},
                      whiskerprops={"color": to_rgba(color, 0.32), "linewidth": 0.4},
                      capprops={"color": to_rgba(color, 0.32), "linewidth": 0.4},
                      medianprops={"color": to_rgba(color, 0.6), "linewidth": 0.5},
                      flierprops={"marker": "o", "markersize": 2.0, "markeredgewidth": 0,
                                  "markerfacecolor": color, "alpha": 0.55}, zorder=2)
            values = advantage.loc[
                advantage.Cohort.eq(cohort) & advantage.Family.eq(family)
            ].set_index("N train").reindex(sizes)
            right.plot(positions, values.Median, color=color, linewidth=1.25, zorder=4)
            counts = wins.loc[
                wins.Cohort.eq(cohort) & wins.Family.eq(family)
            ].set_index("N train").reindex(sizes)
            for position, median, fraction in zip(
                    positions, values.Median, counts["Fraction favoring BLocPRS"]):
                add_win_fraction(right, position, median, fraction, color)
        for column, axis in enumerate([left, right]):
            axis.axhline(0, color="#4A4A4A", linewidth=0.55, linestyle=(0, (1, 1.5)), zorder=0)
            style_axis(axis)
            axis.grid(axis="y", color="#D8D8D8", alpha=0.36, linewidth=0.32)
            axis.set_axisbelow(True)
            axis.set_xlim(-0.4, len(x) - 0.6)
            axis.set_ylim(*(growth_limits if column == 0 else advantage_limits))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.set_xticks(x, size_labels)
            axis.tick_params(axis="both", labelsize=6, length=2.5, pad=1.3)
            axis.text(0, 1.018, "abcd"[row * 2 + column], transform=axis.transAxes,
                      fontsize=8, weight="bold", va="bottom")
            axis.set_ylabel(r"$R^2_N/R^2_{N=2{,}500}-1$ (%)" if column == 0
                            else r"$R^2_{\mathrm{BLocPRS}}/R^2_{\mathrm{MultiPRS}}-1$ (%)",
                            fontsize=7, labelpad=1.0)
        bounds = grid[row, :].get_position(fig)
        fig.text(center_x, bounds.y1 + 0.060, headers[row], fontsize=7, ha="center",
                 va="center", math_fontfamily="dejavusans")
    family_handles = [Line2D([0], [0], color=family_colors[family], linewidth=1.3,
                             label=family.capitalize()) for family in families]
    fig.legend(handles=family_handles, loc="center", bbox_to_anchor=(center_x, 0.941),
               ncol=2, frameon=False, fontsize=6.5, handlelength=1.5, columnspacing=1.0)
    left_bounds, right_bounds = [grid[:, column].get_position(fig) for column in range(2)]
    model_handles = [
        Line2D([0], [0], color="#555555", linestyle="--", marker="o", markerfacecolor="white",
               markersize=3, linewidth=0.9, label="MultiPRS"),
        Line2D([0], [0], color="#555555", marker="s", markersize=3, linewidth=1.25, label="BLocPRS"),
    ]
    fig.legend(handles=model_handles, loc="center",
               bbox_to_anchor=((left_bounds.x0 + left_bounds.x1) / 2, 0.921),
               ncol=2, frameon=False, fontsize=6.5, handlelength=1.5, columnspacing=1.0)
    pie_handle = Line2D([0], [0], linestyle="none", marker="o", markersize=5.5,
                        fillstyle="left", markerfacecolor="#777777", markerfacecoloralt="#F3F3F3",
                        markeredgecolor="#777777", markeredgewidth=0.3,
                        label="Fraction of analyses favoring BLocPRS")
    fig.legend(handles=[pie_handle], loc="center",
               bbox_to_anchor=((right_bounds.x0 + right_bounds.x1) / 2, 0.921),
               frameon=False, fontsize=6.5, handlelength=1.1, handletextpad=0.5)
    fig.supxlabel("Training sample size (N)", x=center_x, y=0.012, fontsize=7)
    return fig


def main():
    args = parse_args(__doc__, collection="sample_size")
    _, growth, paired, advantage, wins = summarize_inputs(load_inputs(args.results))
    with plt.rc_context(style):
        figure = build_figure(growth, paired, advantage, wins)
        save_figure(figure, args.output_dir / "Figure_3.pdf")


if __name__ == "__main__":
    main()
