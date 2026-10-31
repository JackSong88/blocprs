"""Plot the combined benchmark matrices and six-scenario sensitivity summary."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_utils import draw_matrix, save_figure, score_color, score_size, style, style_axis
from .utils import load_benchmark, parse_args


family_colors = {"overall": "#4F4F4F", "continuous": "#2A9D8F", "binary": "#C44E67"}
scenarios = [
    ("ukbb", False, False, "UKB held-out"),
    ("cartagene", False, False, "CaG transfer"),
    ("ukbb", True, False, "non-EUR UKB\nheld-out"),
    ("cartagene", True, False, "non-EUR CaG\ntransfer"),
    ("ukbb", True, True, "non-EUR UKB\nheld-out"),
    ("cartagene", True, True, "non-EUR CaG\ntransfer"),
]
references = ["Nonwindowed MultiPRS", "Best single-source PRS"]


def load_inputs(data_root):
    results = []
    for order, (cohort, non_eur_eval, non_eur_train, _) in enumerate(scenarios):
        data = load_benchmark(data_root, cohort, non_eur_eval, non_eur_train)
        data["Scenario"] = order
        results.append(data)
    return pd.concat(results, ignore_index=True)


def summarize_inputs(data):
    keys = ["Scenario", "Family", "Phenotype"]
    paired = data.pivot(index=keys, columns="Method", values="Value").reset_index()
    paired = paired.loc[np.isfinite(paired["BLocPRS mb1"])
                        & np.isfinite(paired["Nonwindowed MultiPRS"])].copy()
    rows = []
    for reference in references:

        # Apply the 0.01 floor separately to each comparator, avoiding ratios to near-zero R2.
        selected = paired.loc[paired[reference].ge(0.01)].copy()
        selected["Relative gain"] = 100.0 * (selected["BLocPRS mb1"] / selected[reference] - 1.0)
        selected["Reference"] = reference
        rows.append(selected[keys + ["Reference", "Relative gain"]])
    points = pd.concat(rows, ignore_index=True)
    summary = points.groupby(["Scenario", "Family", "Reference"], sort=False).agg(
        Mean=("Relative gain", "mean"), N=("Phenotype", "count"),
    ).reset_index()
    if set(points.Scenario) != set(range(6)):
        raise ValueError("Figure 2 requires all six sensitivity scenarios")
    if points.duplicated(keys + ["Reference"]).any():
        raise ValueError("Duplicate sensitivity comparison")

    # Overall is the equal-weight mean of continuous and binary family means.
    overall = summary.groupby(["Scenario", "Reference"], sort=False).agg(
        Mean=("Mean", "mean"), N=("N", "sum"), Families=("Family", "nunique"),
    ).reset_index()
    if not overall.Families.eq(2).all():
        raise ValueError("Family-balanced summaries require both phenotype families")
    overall["Family"] = "overall"
    return points, pd.concat([summary, overall.drop(columns="Families")], ignore_index=True)




def draw_score_legend(figure):
    ax = figure.add_axes([0.255, 0.875, 0.735, 0.085])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for center, text in [(0.23, "Summary score (% of best-performing method)"),
                         (0.725, "Phenotype score (% of best-performing method)")]:
        ax.text(center, 0.78, text, ha="center", va="center", fontsize=6, fontweight="semibold")
    colors = np.array([score_color("overall", value) for value in np.linspace(0, 1, 256)])
    ax.imshow(colors.reshape(1, 256, 3), extent=(0.135, 0.325, 0.34, 0.50), aspect="auto")
    for x, label in zip([0.135, 0.23, 0.325], ["0%", "50%", "100%"]):
        ax.text(x, 0.13, label, fontsize=5.6, ha="center", va="center")
    for x, value in zip([0.66, 0.725, 0.79], [0.05, 0.50, 1.0]):
        ax.scatter(x, 0.42, s=score_size(value), color=score_color("continuous", value),
                   edgecolor="#3F3F3F", linewidth=0.35)
        ax.text(x, 0.13, f"{value:.0%}", fontsize=5.6, ha="center", va="center")


def draw_sensitivity(figure, points, summary):
    grid = figure.add_gridspec(1, 6, left=0.075, right=0.985, bottom=0.18, top=0.78, wspace=0.12)
    axes = [figure.add_subplot(grid[column]) for column in range(6)]
    labels = []
    for order, (ax, scenario) in enumerate(zip(axes, scenarios)):
        for family, center in [("overall", 2), ("continuous", 1), ("binary", 0)]:
            for reference, offset, alpha in zip(references, [0.14, -0.14], [0.98, 0.72]):
                row = summary.loc[summary.Scenario.eq(order) & summary.Family.eq(family)
                                  & summary.Reference.eq(reference)]
                if len(row) != 1:
                    raise ValueError(f"Missing sensitivity summary: {order}/{family}/{reference}")
                value, y = float(row.Mean.iloc[0]), center + offset
                ax.barh(y, value, height=0.20, color=family_colors[family], alpha=alpha,
                        edgecolor="white", linewidth=0.35, zorder=2)
                artist = ax.text(value / 2, y + (0.11 if offset > 0 else -0.13),
                                 f"{value:+.1f}%", ha="center", fontsize=5.5,
                                 va="bottom" if offset > 0 else "top", color="#444444", zorder=7)
                labels.append((ax, artist, value))
                if family == "overall":
                    continue
                cell = points.loc[points.Scenario.eq(order) & points.Family.eq(family)
                                  & points.Reference.eq(reference), "Relative gain"]
                if len(cell) != int(row.N.iloc[0]):
                    raise ValueError("Sensitivity point/summary cohort mismatch")

                # Horizontal positions are exact; jitter affects only categorical y.
                visible = cell.loc[cell.le(110)]
                jitter = np.resize([0, 0.022, -0.022, 0.044, -0.044], len(visible))
                ax.scatter(visible, y + jitter, s=1.8, color="#444444", alpha=0.45,
                           edgecolor="none", zorder=6)
                overflow = cell.loc[cell.gt(110)].sort_values()
                if len(overflow):

                    # Annotate off-axis values; they remain unchanged in the summary means.
                    ax.scatter(107, y, marker=">", s=5, color="#666666", alpha=0.45, zorder=6)
                    ax.annotate(", ".join(f"{number:.0f}%" for number in overflow), (107, y),
                                xytext=(-2, 0), textcoords="offset points", ha="right",
                                va="center", fontsize=5, color="#777777", zorder=7)
        ax.axvline(0, color="#777777", linewidth=0.5, zorder=1)
        style_axis(ax, grid_axis="x")
        ax.set(xlim=(-50, 110), ylim=(-0.55, 2.55), xticks=[-50, 0, 50, 100], yticks=[2, 1, 0])
        ax.set_title(scenario[3], fontsize=5.8, pad=3, linespacing=0.95)
        ax.set_yticklabels(["Overall", "Continuous", "Binary"] if order == 0 else [])
        ax.tick_params(axis="x", labelsize=5.3)
        ax.tick_params(axis="y", labelsize=5.5, left=order == 0)
    for start, end, text in [(0, 3, "Full UKB training"), (4, 5, "non-EUR UKB training")]:
        left, right = axes[start].get_position().x0, axes[end].get_position().x1
        figure.text((left + right) / 2, 0.905, text, ha="center", fontsize=6.6, fontweight="semibold")
    figure.text(0.028, 0.995, "C", fontsize=8.5, fontweight="bold", va="center")
    figure.text(0.055, 0.995, "BLocPRS - 1-Mb (primary)", fontsize=7.4, fontweight="semibold", va="center")
    handles = [Line2D([], [], color=color, linewidth=4) for color in ["#555555", "#A5A5A5"]]
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=2,
                          markerfacecolor="#666666", markeredgewidth=0, alpha=0.45))
    figure.legend(handles, ["vs MultiPRS", "vs best single-source PRS", "Phenotype-panel analysis"],
                  frameon=False, loc="center", ncol=3, fontsize=6, bbox_to_anchor=(0.55, 0.995))
    figure.supxlabel(r"$R^2_{\mathrm{BLocPRS}} / R^2_{\mathrm{reference}} - 1$ (%)", fontsize=7.5, y=0.02)
    return labels


def build_figure(data, points, summary):
    fig = plt.figure(figsize=(183 / 25.4, 156 / 25.4), facecolor="white")
    matrix, sensitivity = fig.subfigures(2, 1, height_ratios=[4.83, 2.45], hspace=0.018)
    grid = matrix.add_gridspec(2, 4, width_ratios=[1.50, 0.58, 0.07, 7.6],
                              hspace=0.35, wspace=0, left=0.028, right=0.985, bottom=0.025, top=0.79)
    for order, (label, title) in enumerate([
            ("A", "Held-out UK Biobank"), ("B", r"UK Biobank $\rightarrow$ CARTaGENE transfer")]):
        row_grid = [grid[order, column] for column in range(4)]
        draw_matrix(matrix, row_grid, data.loc[data.Scenario.eq(order)], label, title)
    draw_score_legend(matrix)
    labels = draw_sensitivity(sensitivity, points, summary)

    # Move labels that cross the zero line without changing their displayed values.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, artist, value in labels:
        box = artist.get_window_extent(renderer=renderer)
        zero = ax.transData.transform((0, 0))[0]
        if box.x0 <= zero <= box.x1:
            artist.set_x(1.6 if value >= 0 else -1.6)
            artist.set_ha("left" if value >= 0 else "right")
    return fig


def main():
    args = parse_args(__doc__)
    data = load_inputs(args.results)
    points, summary = summarize_inputs(data)
    with plt.rc_context(style):
        figure = build_figure(data, points, summary)
        save_figure(figure, args.output_dir / "Figure_2.pdf")


if __name__ == "__main__":
    main()
