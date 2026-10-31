"""Pooled-sex training evaluated in full, male, and female participant strata."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..utils import load_sex_results, parse_args
from ..plot_utils import draw_comparison, save_figure, style, style_axis


methods = ["Male PRS", "Female PRS", "Nonwindowed MultiPRS", "BLocPRS mb1", "BLocPRS mb3", "BLocPRS mb5"]
method_labels = ["Male PRS", "Female PRS", "MultiPRS", "BLocPRS (1 Mb)", "BLocPRS (3 Mb)", "BLocPRS (5 Mb)"]
colors = ["#72A58B", "#D9776A", "#6BAED6", "#6A51A3", "#9279B8", "#B5A3CE"]


def draw_cell(ax, frame, tests, limits, first):
    indexed = frame.set_index("method").loc[methods]
    x = np.arange(len(methods))*.65
    values, errors = indexed.value.to_numpy(), indexed.err.to_numpy()
    ax.bar(x, values, width=.50, color=colors, edgecolor="#666666", lw=.18, zorder=3)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="#363636", elinewidth=.48, capsize=0, zorder=4)
    ax.set(xlim=(-.55, x[-1]+.55), ylim=limits)
    style_axis(ax)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set_xticks([])
    ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(*limits)
    ax.set_yticks([tick for tick in ticks if limits[0] <= tick <= limits[1]])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-10 else f"{value:.2f}"))
    ax.tick_params(direction="out", length=1.8, width=.45, labelsize=5.7,
                   left=first, labelleft=first, bottom=False, top=False, right=False)
    ax.spines["left"].set_visible(first)
    row = frame.iloc[0]
    selected = tests.loc[tests.scenario.eq(row.scenario) & tests.phenotype.eq(row.phenotype)
                         & tests.stratum.eq(row.stratum)].set_index("baseline")

    # This supplement compares the best observed sex source, unlike main Figure 7.
    strongest = indexed.loc[methods[:2], "value"].idxmax()
    top, span = float(np.max(values+errors)), limits[1]-limits[0]
    level = 0
    for baseline in ("Nonwindowed MultiPRS", strongest):
        p = float(selected.loc[baseline, "p_wald_two_sided"])
        if p < .05:
            draw_comparison(ax, x[methods.index(baseline)], x[methods.index("BLocPRS mb1")],
                            p, top+(.065+.19*level)*span, .025*span)
            level += 1


def build_figure(data, tests):
    phenotypes = ["WHR_SEX", "URT_SEX", "LOG_CRTN_SEX", "LOG_TST_SEX"]
    labels = ["WHR", "Urate", "Creatinine\u2020", "Testosterone\u2020"]
    height, left, right = 145, .105, .980
    center, width = (left+right)/2, (right-left)/4
    start, bottom, gap, header, row_gap = 10, 9, 7, 14.5, 3
    block_height = (height-start-bottom-gap)/2
    plot_height = block_height-header
    row_height = (plot_height-2*row_gap)/3
    fig = plt.figure(figsize=(183/25.4, height/25.4))
    low = min(0, float((data.value-data.err).min()))
    high = float((data.value+data.err).max())
    span = max(high-low, .02)
    limits = (low-(.03*span if low < 0 else 0), high+.85*span)
    fig.text(center, 1-3/height, "Full UKB training", fontsize=8, ha="center", va="top")
    fig.legend([Patch(facecolor=color, edgecolor="none") for color in colors], method_labels,
               loc="lower center", bbox_to_anchor=(.5, 1.5/height), ncol=len(methods), frameon=False,
               fontsize=6.2, columnspacing=1, handlelength=1.4, handletextpad=.45)
    for block, (scenario, title) in enumerate((("ukbb_native", "Held-out UK Biobank"),
            ("ukbb_to_cartagene", r"UK Biobank $\rightarrow$ CARTaGENE transfer"))):
        top = start+block*(block_height+gap)
        fig.text(.018, 1-top/height, "ab"[block], fontsize=8.5, fontweight="bold", va="top")
        fig.text(center, 1-top/height, title, fontsize=7.4, va="top", ha="center")
        band_y = 1-(top+7)/height
        fig.add_artist(FancyBboxPatch((left, band_y), right-left, 3/height,
                       boxstyle="round,pad=0,rounding_size=0.002", transform=fig.transFigure,
                       facecolor="#a2cdb7", edgecolor="none", zorder=0))
        fig.text(center, band_y+1.5/height, "Sex-specific", ha="center", va="center", fontsize=5.3, color="#202629")
        for column, label in enumerate(labels):
            fig.text(left+(column+.465)*width, 1-(top+8.5)/height, label, fontsize=6.5, ha="center", va="top")
        plot_top = 1-(top+header)/height
        for row, stratum in enumerate(("All", "Male", "Female")):
            axis_top = plot_top-row*(row_height+row_gap)/height
            fig.text(.05, axis_top-row_height/(2*height), "Full" if stratum == "All" else stratum,
                     fontsize=6, rotation=90, ha="center", va="center")
            for column, phenotype in enumerate(phenotypes):
                ax = fig.add_axes([left+column*width, axis_top-row_height/height, width*.93, row_height/height])
                if scenario == "ukbb_to_cartagene" and phenotype == "LOG_TST_SEX":
                    ax.set(xlim=(-.55, 5*.65+.55), ylim=limits)
                    style_axis(ax)
                    ax.spines["bottom"].set_position(("data", 0))
                    ax.spines["left"].set_visible(False)
                    ax.set(xticks=[], yticks=[])
                    ax.text(.5, .5, "--", transform=ax.transAxes, fontsize=12, color="#777777", ha="center", va="center")
                    continue
                cell = data.loc[data.scenario.eq(scenario) & data.phenotype.eq(phenotype) & data.stratum.eq(stratum)]
                draw_cell(ax, cell, tests, limits, column == 0)
        fig.text(.018, plot_top-plot_height/(2*height), r"Incremental $R^2$", fontsize=6.8,
                 rotation=90, ha="center", va="center")
    return fig


def main():
    args = parse_args(__doc__, supplementary=True)
    values, tests = load_sex_results(args.results)
    with plt.rc_context(style):
        save_figure(build_figure(values, tests), args.output_dir / "suppl_sex_specific.pdf")


if __name__ == "__main__":
    main()
