"""Sensitivity of genome-wide MultiPRS to expert-weight regularization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from ..utils import core_phenotypes, metric_tokens, one_row, parse_args, read_results, select_results
from ..plot_utils import family_colors, save_figure, style, style_axis


reference = "MultiPRS-OLS"
methods = ("MultiPRS-Ridge", "MultiPRS-Lasso", "MultiPRS-ElasticNet", "MultiPRS-Nonnegative")
method_labels = ("Ridge", "Lasso", "Elastic net", "Nonnegative")
cohorts = {"ukbb": "Held-out UKB", "cartagene": "UKB to CaG transfer"}
analyses = core_phenotypes + [(name, "continuous") for name in (
    "LDL_ADJ_MA", "TC_ADJ_MA", "DBP_ADJ_MA", "SBP_ADJ_MA")]


# Compare every regularized fit to unregularized MultiPRS on the same evaluation set.
def load_results(path):
    data = read_results(path)
    records = []
    tokens = ("ridge", "lasso", "elastic_net", "nonnegative")
    for cohort in cohorts:
        for phenotype, family in analyses:

            if cohort == "cartagene" and phenotype in {"LOG_TST_SEX", "DEM_MT"}:
                continue

            frame = select_results(data, phenotype=phenotype, cohort=cohort, group="all",
                                   train_group="all", train_size="full", panel="main",
                                   covariate_set="standard", metric=metric_tokens[family], window_mb=0)
            baseline = one_row(frame, method="multiprs")

            if baseline.value <= 1e-8:
                raise ValueError(f"Unusable MultiPRS denominator: {phenotype}/{cohort}")

            for method, token in zip(methods, tokens):
                row = one_row(frame, method=token)

                if row.n != baseline.n:
                    raise ValueError("Regularization methods evaluate different participant counts.")

                records.append(dict(cohort=cohort, phenotype=phenotype, family=family, method=method,
                                    relative_change_pct=100 * (row.value / baseline.value - 1)))
    return pd.DataFrame(records)


# Bars are means of phenotype-specific percentages; points retain each comparison.
def build_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 100 / 25.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.21, top=.85, wspace=.12)
    limits = (min(-5, data.relative_change_pct.min()*1.05), max(5, data.relative_change_pct.max()*1.05))
    for index, (ax, (cohort, title)) in enumerate(zip(axes, cohorts.items())):
        style_axis(ax, grid_axis="y")
        for family, offset, marker in (("continuous", -.15, "o"), ("binary", .15, "s")):
            color = family_colors[family]
            for x, method in enumerate(methods):
                frame = data.loc[data.cohort.eq(cohort) & data.family.eq(family)
                                 & data.method.eq(method)].sort_values("phenotype")
                values = frame.relative_change_pct.to_numpy()
                ax.bar(x+offset, values.mean(), width=.24, color=color, alpha=.82,
                       edgecolor="#666666", linewidth=.25, zorder=3)
                ax.scatter(x+offset+np.linspace(-.085, .085, len(values)), values, s=5.5, marker=marker,
                           color=color, alpha=.48, edgecolors="white", linewidths=.15, zorder=4)
        ax.axhline(0, color="#4A4A4A", linewidth=.45, zorder=2)
        ax.set(xlim=(-.5, 3.5), ylim=limits)
        ax.set_xticks(range(4), method_labels)
        if limits == (-5, 5):
            ax.set_yticks((-5, -2.5, 0, 2.5, 5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.tick_params(labelsize=6.5, left=index == 0, labelleft=index == 0)
        ax.spines["left"].set_visible(index == 0)
        box = ax.get_position()
        fig.text(box.x0, .875, "ab"[index], fontsize=8, fontweight="bold", va="bottom")
        fig.text((box.x0+box.x1)/2, .875, title, fontsize=7, ha="center", va="bottom")
    axes[0].set_ylabel(r"Relative change in $R^2$ vs MultiPRS (%)", fontsize=6.8, labelpad=6)
    handles = [Patch(facecolor=color, edgecolor="none", label=family.capitalize())
               for family, color in family_colors.items()]
    handles += [Patch(facecolor="#A6A6A6", edgecolor="none", label="Bar: mean"),
                Line2D([], [], marker="o", markersize=2.4, linestyle="none", color="#777777",
                       label="Point: phenotype-panel analysis")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .06), ncol=4,
               frameon=False, fontsize=5.7, columnspacing=1.4, handlelength=1.6, handletextpad=.45)
    return fig


# Export the sensitivity comparison from saved evaluations.
def main():
    args = parse_args(__doc__, supplementary=True, collection="regularization")
    data = load_results(args.results)
    with plt.rc_context(style):
        save_figure(build_figure(data), args.output_dir / "suppl_regularization.pdf")


if __name__ == "__main__":
    main()
