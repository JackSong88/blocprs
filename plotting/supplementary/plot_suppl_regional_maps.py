"""Detailed chromosome maps of coefficients, PRS SD, contributions and training GWAS."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..utils import load_regional_panel, parse_args, regional_specs
from ..plot_utils import (
    nice_limit, regional_gwas_track, regional_heatmap, save_figure, source_colors, style, style_axis,
)


# Place coefficient, input-SD and contribution tracks above the two GWAS tracks.
def build_figure(panel):
    height = 181
    fig = plt.figure(figsize=(183 / 25.4, height / 25.4))
    axes = []
    top = 12
    for row, (track_height, gap) in enumerate(zip((15, 15, 15, 15, 15, 22, 22),
                                                (2.5, 6, 6, 2.5, 7, 4, 0))):
        ax = fig.add_axes([.105, 1 - (top + track_height) / height, .810, track_height / height],
                          sharex=axes[0] if axes else None)
        top += track_height + gap
        axes.append(ax)
        style_axis(ax)
        ax.set_xlim(0, panel["ends"][-1])
        ax.set_xticks([tick for tick in panel["ticks"] if tick <= panel["ends"][-1]])
        ax.tick_params(axis="x", bottom=row == 6, labelbottom=row == 6, pad=2, labelsize=7)
        ax.tick_params(axis="y", labelsize=7, pad=2)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("regional_coefficient", ["#355F8D", "#F7F7F5", "#B8424C"])

    # Effective and adjustment coefficients share a zero-centered color scale.
    limit = nice_limit(max(np.abs(panel["coefficients"][i]).max() for i in (0, 2)))
    norm = mpl.colors.Normalize(-limit, limit)
    coefficient_mesh = regional_heatmap(axes[0], panel, panel["coefficients"][0], cmap, norm, fontsize=7)
    regional_heatmap(axes[1], panel, panel["coefficients"][2], cmap, norm, fontsize=7)

    # PRS SD is nonnegative and uses its own scale, separate from signed contributions.
    sd_max = nice_limit(panel["sd"].max())
    sd_cmap = mpl.colors.LinearSegmentedColormap.from_list("training_prs_sd", ["#F7F7F5", "#355F8D"])
    sd_mesh = regional_heatmap(axes[2], panel, panel["sd"], sd_cmap,
                               mpl.colors.Normalize(0, sd_max), fontsize=7)
    contribution_limit = nice_limit(max(np.abs(panel["contributions"][i]).max() for i in (0, 2)))
    norm = mpl.colors.Normalize(-contribution_limit, contribution_limit)
    contribution_mesh = regional_heatmap(axes[3], panel, panel["contributions"][0], cmap, norm, fontsize=7)
    regional_heatmap(axes[4], panel, panel["contributions"][2], cmap, norm, fontsize=7)
    groups = (("Female",), ("Male",)) if panel["key"] == "creatinine" else (
        ("EUR",), tuple(source for source in panel["sources"] if source != "EUR"))
    labels = ("Female", "Male") if panel["key"] == "creatinine" else ("EUR", "Non-EUR")
    for ax, group, label in zip(axes[5:], groups, labels):
        regional_gwas_track(ax, panel, group, label, detailed=True)

    # Highlight the same genomic landmarks as the compact maps.
    for left, right, name in panel["loci"]:
        for ax in axes:
            ax.axvspan(left, right, facecolor="#D6A84F", edgecolor="#8A6B2F",
                       linewidth=.35, alpha=.09, zorder=5)
        center = (left + right) / 2
        axes[0].plot([center, center], [1, 1.065], transform=axes[0].get_xaxis_transform(),
                     clip_on=False, color="#8A6B2F", linewidth=.55)
        axes[0].text(center, 1.10, name, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=7, fontweight="semibold", fontstyle="normal" if name == "MHC" else "italic",
                     color="#6F5628")
    for first, last, label in ((0, 0, "Effective\ncoefficient"), (1, 1, "Locus-specific\nadjustment coefficient"),
                                (2, 2, "Window PRS\nstandard deviation"), (3, 3, "Effective\ncontribution"),
                                (4, 4, "Locus-specific\nadjustment contribution"),
                                (5, 6, "GWAS association\n" + r"$-\log_{10}(P)$")):
        center = (axes[first].get_position().y1 + axes[last].get_position().y0) / 2
        fig.text(.025, center, label, rotation=90, ha="center", va="center", fontsize=7, linespacing=1.15)
    for artist, first, last, ticks in ((coefficient_mesh, 0, 1, [-limit, 0, limit]),
                                       (sd_mesh, 2, 2, [0, sd_max]),
                                       (contribution_mesh, 3, 4, [-contribution_limit, 0, contribution_limit])):
        cax = fig.add_axes([.927, axes[last].get_position().y0, .009,
                            axes[first].get_position().y1 - axes[last].get_position().y0])
        colorbar = fig.colorbar(artist, cax=cax, ticks=ticks)
        colorbar.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:g}"))
        colorbar.ax.tick_params(labelsize=7, length=1.8, pad=1.2)
        colorbar.outline.set_linewidth(.4)
    axes[-1].set_xlabel(f"Position on chromosome {panel['chromosome']} (Mb)", fontsize=7, labelpad=3)
    fig.text(.51, .985, f"{panel['title']} - chromosome {panel['chromosome']}", fontsize=7, ha="center", va="top")
    for upper, lower, symbol in ((1, 2, "x"), (2, 3, "=")):
        center = (axes[upper].get_position().y0 + axes[lower].get_position().y1) / 2
        fig.text(.51, center, symbol, fontsize=10, ha="center", va="center", color="#555555")
    handles = [Line2D([], [], marker="o", linestyle="none", color=source_colors[source], markersize=3)
               for source in panel["sources"]]
    fig.legend(handles, panel["sources"], title="PRS source", loc="lower center", bbox_to_anchor=(.51, .018),
               ncol=len(handles), frameon=False, fontsize=7, title_fontsize=7, handletextpad=.35, columnspacing=1.4)
    return fig


# Export one detailed map for each prespecified phenotype/chromosome pair.
def main():
    args = parse_args(__doc__, supplementary=True, interpretation=True, regional=True)
    with plt.rc_context(style):
        for spec in regional_specs:
            panel = load_regional_panel(args.data_root, spec, args.analysis_dir, args.gwas_template)
            save_figure(build_figure(panel), args.output_dir / f"suppl_regional_map_{spec['key']}.pdf")


if __name__ == "__main__":
    main()
