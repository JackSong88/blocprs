"""Training-scale contributions, local-gain recovery, and source-composition shifts."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .utils import load_local_recovery, load_regional_interpretation, parse_args
from .plot_utils import (contribution_colors, draw_signed, ordered_sources, reference_bins,
                         save_figure, style, style_axis)


phenotypes = ["LDL_MA", "ASTHMA_MA", "LOG_CRTN_SEX", "HEIGHT_MA"]
labels = {"LDL_MA": "LDL-C", "ASTHMA_MA": "Asthma", "LOG_CRTN_SEX": "Creatinine\u2020", "HEIGHT_MA": "Height"}
colors = dict(zip(phenotypes, ["#0072B2", "#D55E00", "#009E73", "#8C78A8"]))
markers = dict(zip(phenotypes, ["o", "^", "D", "s"]))
marker_sizes = dict(zip(phenotypes, [3.8, 4.1, 3.5, 2.6]))


# The random band pools phenotype-by-draw values; it is an IQR, not a confidence interval.
def draw_recovery(ax, recovery, band, scenario):
    rows = recovery.loc[recovery.Scenario.eq(scenario)]
    band = band.loc[band.Scenario.eq(scenario)].sort_values("RequestedFractionPct")
    x = np.arange(8)
    ax.fill_between(x, band.q25.to_numpy(), band.q75.to_numpy(), color="#8F8F8F", alpha=.20, linewidth=0)
    ax.plot(x, band["median"], color="#666666", ls="--", lw=.95)
    for phenotype in phenotypes:
        values = rows.loc[rows.Phenotype.eq(phenotype)].sort_values("RequestedFractionPct")
        ax.plot(x, values.LocalGainRecoveryPct, color=colors[phenotype], marker=markers[phenotype],
                ms=marker_sizes[phenotype], markeredgewidth=0, lw=.95 if phenotype == "HEIGHT_MA" else 1.15)
    extrema = np.r_[rows.LocalGainRecoveryPct, band.q25, band.q75]
    ax.set_ylim(min(-5, extrema.min()-3), max(105, extrema.max()+3))
    ax.axhline(100, color="#999999", ls=":", lw=.6)
    ax.set(xticks=x, xticklabels=["0", "1", "2.5", "5", "10", "25", "50", "100"],
           xlim=(-.15, len(x)-.85), yticks=[0, 25, 50, 75, 100])
    style_axis(ax)
    ax.tick_params(labelsize=7)
    for position, caption in ((0, "Genome-wide\nonly"), (7, "Full\nBLocPRS")):
        ax.annotate(caption, xy=(position, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -13), textcoords="offset points", ha="center", va="top",
                    fontsize=6.2, color="#444444", annotation_clip=False)


# Diamonds are reference-mass-weighted means; boxes retain 5th/95th-percentile whiskers.
def draw_shares(ax, rows, rng, limit):
    sources = ordered_sources(rows.expert_label)
    for position, source in enumerate(sources):
        group = rows.loc[rows.expert_label.eq(source)]
        values, color = group["shift"].to_numpy(), contribution_colors[source]
        box = ax.boxplot([values], positions=[position], widths=.48, whis=(5, 95), patch_artist=True,
                         showfliers=False, boxprops={"facecolor": color, "edgecolor": color, "linewidth": .68},
                         whiskerprops={"color": color, "linewidth": .68}, capprops={"color": color, "linewidth": .68},
                         medianprops={"color": "#5A5A5A", "linewidth": .92})
        box["boxes"][0].set_alpha(.16)
        ax.scatter(position+rng.uniform(-.14, .14, len(values)), values, s=4.8,
                   color=color, alpha=.40, edgecolors="none", rasterized=True, zorder=3)
        ax.scatter([position], [np.average(values, weights=group.G)], marker="D", s=11.5,
                   facecolor="white", edgecolor="#333333", linewidth=.48, zorder=5)
    ax.text(.025, .96, labels[rows.phenotype.iloc[0]], transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.axhline(0, color="#888888", lw=.6)
    ax.set(xticks=np.arange(len(sources)), xticklabels=sources, xlim=(-.55, len(sources)-.45),
           ylim=(-1.06*limit, 1.06*limit))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    style_axis(ax)
    ax.spines["left"].set_color("#B8BABC")
    ax.spines["left"].set_linewidth(.45)
    ax.tick_params(labelsize=7, length=3, width=.65)


def build_figure(ratios, top, recovery, band):
    height = 162
    fig = plt.figure(figsize=(183/25.4, height/25.4))

    def y(mm):
        return 1-mm/height

    core = ratios.loc[~ratios.phenotype.str.contains("_ADJ_")]

    # Average phenotype-level ratios within each bin, rather than pooling all windows.
    magnitude = core.groupby("bin").ratio.mean().reindex(reference_bins)
    ax = fig.add_axes([.085, y(50), .18, 33/height])
    ax.plot(np.arange(7), magnitude, color="#6A51A3", marker="o", lw=1.2, ms=3.5)
    ax.set(xticks=np.arange(7), xticklabels=reference_bins, ylim=(0, magnitude.max()*1.16))
    ax.set_ylabel("Adjustment contribution\n(% of reference contribution)", fontsize=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_axis(ax)
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    fig.text(.175, y(14), "Locus-specific adjustment\nacross reference-ranked windows", ha="center", fontsize=7)
    fig.text(.175, y(63), "Genome-wide reference contribution\n(window percentile)", ha="center", fontsize=7)

    axes = []
    for left, width, scenario, title in ((.35, .285, "ukbb_native", "Held-out UK Biobank"),
                                        (.70, .28, "full_cag_transfer", r"UKB $\rightarrow$ CaG transfer")):
        ax = fig.add_axes([left, y(50), width, 29/height])
        draw_recovery(ax, recovery, band, scenario)
        ax.set_title(title, fontsize=7, pad=5)
        axes.append(ax)
    limits = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_ylim(limits)
    axes[0].set_ylabel("Prediction gain recovered (%)", fontsize=7, labelpad=2)
    axes[1].tick_params(axis="y", labelleft=False)
    handles = [Line2D([], [], color=colors[p], marker=markers[p], ms=marker_sizes[p], markeredgewidth=0,
                      lw=.95 if p == "HEIGHT_MA" else 1.15, label=labels[p]) for p in phenotypes]
    controls = [Patch(facecolor="#8F8F8F", alpha=.20, label="Random-window IQR"),
                Line2D([], [], color="#666666", ls="--", lw=.95, label="Random-window median")]
    legends = [fig.legend(handles=group, loc="upper left", bbox_to_anchor=(0, y(9)), ncol=len(group),
                          frameon=False, borderaxespad=0, fontsize=7, handlelength=.9,
                          columnspacing=.55, handletextpad=.25) for group in (handles, controls)]

    # Measure rendered text so the two legend blocks can be centered together.
    fig.canvas.draw()
    widths = [legend.get_window_extent().width/fig.bbox.width for legend in legends]
    gap = 10/(72*fig.get_figwidth())
    left = .665-(sum(widths)+gap)/2
    legends[0].set_bbox_to_anchor((left, y(9)))
    legends[1].set_bbox_to_anchor((left+widths[0]+gap, y(9)))
    fig.text(.665, y(62), "Top-ranked windows retaining locus-specific adjustments (%)", ha="center", fontsize=7)

    fig.text(.3325, y(71), "Source contributions in the top 1% of windows", ha="center", fontsize=7)
    fig.text(.825, y(75), "Composition shifts in the\ntop 1% of windows", ha="center", fontsize=7)
    for left, title in ((.085, "Genome-wide reference"), (.37, "Full BLocPRS")):
        fig.text(left+.105, y(75), title, ha="center", fontsize=7)
    selected = top.loc[top.phenotype.isin(phenotypes[:3])]
    limit = max(5, np.ceil(selected["shift"].abs().max()/5)*5)

    # Matching seeds show the same window jitter in the paired signed panels.
    rngs = [np.random.default_rng(20260817) for _ in range(3)]
    for index, phenotype in enumerate(phenotypes[:3]):
        rows = selected.loc[selected.phenotype.eq(phenotype)]
        for left, column, rng in zip((.085, .37), ("reference", "effective"), rngs[:2]):
            ax = fig.add_axes([left, y(99+27*index), .21, 20/height])
            draw_signed(ax, rows, column, rng)
            ax.text(.97, .96, labels[phenotype], transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax = fig.add_axes([.685, y(99+27*index), .28, 20/height])
        draw_shares(ax, rows, rngs[2], limit)
    fig.text(.015, y(116), "Signed prediction-scale contribution", rotation=90, va="center", fontsize=7)
    fig.text(.625, y(116), "Source-composition change (percentage points)", rotation=90, va="center", fontsize=7)
    for center in (.3325, .825):
        fig.text(center, y(161), "PRS source", ha="center", fontsize=7)
    for letter, left, top_mm in (("a", .015, 7), ("b", .305, 7), ("c", .025, 67), ("d", .605, 67)):
        fig.text(left, y(top_mm), letter, fontsize=8, fontweight="bold", va="top")
    return fig


def main():
    args = parse_args(__doc__, interpretation=True)
    _, ratios, top = load_regional_interpretation(args.data_root, args.analysis_dir)
    recovery, band = load_local_recovery(args.data_root, args.analysis_dir)
    with plt.rc_context(style):
        save_figure(build_figure(ratios, top, recovery, band), args.output_dir / "Figure_6.pdf")


if __name__ == "__main__":
    main()
