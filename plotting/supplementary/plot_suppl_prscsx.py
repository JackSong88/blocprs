"""Compare combinations of the fixed PRS-CSx-derived ancestry components."""

from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..plot_utils import compact_colors, compact_labels, cohort_labels, save_figure, style, style_axis
from ..utils import compact_scenarios, metric_tokens, method_row, parse_args, read_results, select_results


phenotypes = {
    "LOG_BMI_MA": ("BMI\u2020", "continuous", ["PGS005200", "PGS005201", "PGS005202", "PGS005203", "PGS005204"]),
    "ASTHMA_MA": ("Asthma", "binary", ["PGS005121", "PGS005128", "PGS005134", "PGS005140"]),
    "T2D_MA": ("Type 2 diabetes", "binary", ["PGS005243", "PGS005244", "PGS005245", "PGS005246", "PGS005247"]),
}
methods = ["Best component", "MultiPRS", "BLocPRS mb1", "BLocPRS mb3", "BLocPRS mb5"]


# The best source is selected separately in each evaluation cohort.
def load_results(path):
    data = read_results(path)
    rows = []
    for scenario, cohort in zip(compact_scenarios, ["ukbb", "cartagene"]):
        for phenotype, (_, family, components) in phenotypes.items():
            frame = select_results(data, cohort=cohort, phenotype=phenotype, group="all",
                                   train_group="all", train_size="full", panel="prscsx",
                                   covariate_set="standard", metric=metric_tokens[family])
            singles = frame.loc[frame.method.eq("single")]

            if set(singles.source) != set(components):
                raise ValueError(f"Incomplete PRS-CSx source panel: {phenotype}")

            selected = [method_row(frame, "single"), method_row(frame, "multiprs")]
            selected += [method_row(frame, "blocprs", mb) for mb in (1, 3, 5)]

            if len({row.n for row in selected}) != 1:
                raise ValueError("PRS-CSx methods evaluate different participant counts.")

            # The 0.01 floor applies to the MultiPRS denominator, not to each method.
            reference = selected[1].value
            for method, row in zip(methods, selected):
                relative = 100 * (row.value / reference - 1) if reference >= .01 else np.nan
                rows.append(dict(scenario=scenario, phenotype=phenotype, method=method,
                                 value=row.value, reference=reference, relative_change=relative))
    return pd.DataFrame(rows)


def build_figure(data):
    figure, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 100 / 25.4), sharey=True)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.205, top=0.85, wspace=0.12)
    finite = data.relative_change.dropna()
    low, high = (min(0, finite.min()), max(0, finite.max())) if len(finite) else (0, 0)
    padding = max(3, 0.1 * max(10, high - low))
    limits = (5 * floor((low - padding) / 5), 5 * ceil((high + padding) / 5))
    positions = np.arange(3)
    displayed = [methods[0], *methods[2:]]
    colors = [compact_colors[0], *compact_colors[2:]]
    for panel, (ax, scenario, title) in enumerate(zip(axes, compact_scenarios, cohort_labels)):
        selected = data.loc[data.scenario.eq(scenario)].set_index(["phenotype", "method"])
        style_axis(ax, grid_axis="y")
        for index, (method, color) in enumerate(zip(displayed, colors)):
            heights = np.array([selected.loc[(phenotype, method), "relative_change"] for phenotype in phenotypes])
            finite = np.isfinite(heights)
            ax.bar((positions + (index - 1.5) * 0.145)[finite], heights[finite], width=0.132,
                   color=color, edgecolor="#666666", linewidth=0.18, zorder=3)
        for position, phenotype in enumerate(phenotypes):
            if selected.loc[(phenotype, "MultiPRS"), "reference"] < 0.01:
                ax.text(position, 0, "N/A", fontsize=5.7, ha="center", va="bottom")
        ax.axhline(0, color="#4A4A4A", linestyle=":", linewidth=0.55, zorder=4)
        ax.set(xlim=(-0.48, 2.48), ylim=limits, xticks=positions,
               xticklabels=[spec[0] for spec in phenotypes.values()])
        ax.tick_params(length=2, labelsize=6.8, left=panel == 0, labelleft=panel == 0)
        ax.spines["left"].set_visible(panel == 0)
        box = ax.get_position()
        figure.text(box.x0, 0.875, "AB"[panel], fontsize=8.5, fontweight="bold", va="bottom")
        figure.text((box.x0 + box.x1) / 2, 0.875, title, fontsize=7, ha="center", va="bottom")
    axes[0].set_ylabel(r"Relative change in $R^2$ vs MultiPRS (%)", fontsize=6.8, labelpad=6)
    handles = [Patch(facecolor=colors[0], label="Best ancestry-specific component"),
               Line2D([0], [0], color="#4A4A4A", linestyle=":", linewidth=0.55, label="MultiPRS (reference)")]
    handles += [Patch(facecolor=color, label=label) for color, label in zip(colors[1:], compact_labels[2:])]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=5,
                  fontsize=6.8, frameon=False, columnspacing=1, handlelength=1.4, handletextpad=0.45)
    return figure


def main():
    args = parse_args(__doc__, supplementary=True, collection="prscsx")
    data = load_results(args.results)
    with plt.rc_context(style):
        print(save_figure(build_figure(data), args.output_dir / "suppl_prscsx.pdf"))


if __name__ == "__main__":
    main()
