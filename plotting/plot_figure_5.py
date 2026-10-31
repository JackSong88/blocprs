"""Regional coefficients and training-GWAS associations for LDL, asthma and creatinine."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .utils import load_regional_panel, parse_args, regional_specs
from .plot_utils import (
    nice_limit, regional_gwas_track, regional_heatmap, save_figure, source_colors, style, style_axis,
)


# Use identical scales for effective and centered local coefficients within a trait.
def build_figure(panels):
    fig = plt.figure(figsize=(183 / 25.4, 130 / 25.4))
    grid = fig.add_gridspec(4, 3, height_ratios=[.68, .68, 1.18, 1.18],
                            left=.075, right=.952, top=.900, bottom=.135, wspace=.265, hspace=.180)
    axes = np.empty((4, 3), dtype=object)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("regional_coefficient", ["#355F8D", "#F7F7F5", "#B8424C"])
    for column, panel in enumerate(panels):
        for row in range(4):
            axes[row, column] = fig.add_subplot(grid[row, column], sharex=axes[0, column] if row else None)
            ax = axes[row, column]
            style_axis(ax)
            ax.set_xlim(0, panel["ends"][-1])
            ax.set_xticks(panel["ticks"])
            ax.tick_params(axis="x", labelbottom=row == 3, bottom=row == 3, pad=1.4)
            ax.tick_params(axis="y", labelsize=6)

        # Effective and adjustment coefficients share a zero-centered scale within this phenotype.
        limit = nice_limit(max(np.abs(panel["coefficients"][i]).max() for i in (0, 2)))
        norm = mpl.colors.Normalize(-limit, limit)
        mesh = regional_heatmap(axes[0, column], panel, panel["coefficients"][0], cmap, norm)
        regional_heatmap(axes[1, column], panel, panel["coefficients"][2], cmap, norm)

        # Non-EUR overlays retain separate ancestry-specific GWAS points, not a pooled association test.
        groups = (("Female",), ("Male",)) if panel["key"] == "creatinine" else (
            ("EUR",), tuple(source for source in panel["sources"] if source != "EUR"))
        labels = ("Female", "Male") if panel["key"] == "creatinine" else ("EUR", "Non-EUR")
        for ax, group, label in zip(axes[2:, column], groups, labels):
            regional_gwas_track(ax, panel, group, label)

        # Locus annotations are landmarks, not inferred causal intervals.
        for left, right, name in panel["loci"]:
            for ax in axes[:, column]:
                ax.axvspan(left, right, facecolor="#D6A84F", edgecolor="#8A6B2F",
                           linewidth=.35, alpha=.09, zorder=5)
            center = (left + right) / 2
            ax = axes[0, column]
            ax.plot([center, center], [1, 1.027], transform=ax.get_xaxis_transform(),
                    color="#8A6B2F", linewidth=.55, clip_on=False)
            ax.text(center, 1.04, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.5, fontweight="semibold", color="#6F5628",
                    fontstyle="normal" if name == "MHC" else "italic")
        upper, lower = axes[0, column].get_position(), axes[1, column].get_position()
        cax = fig.add_axes([upper.x1 + .0045, lower.y0, .006, upper.y1 - lower.y0])
        colorbar = fig.colorbar(mesh, cax=cax, ticks=[-limit, 0, limit])
        colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda value, _: "0" if np.isclose(value, 0, atol=5e-4) else f"{value:.2f}"))
        colorbar.outline.set_linewidth(.4)
        colorbar.ax.tick_params(labelsize=5.8, length=1.6, pad=1)
        axes[-1, column].set_xlabel(f"Position on chromosome {panel['chromosome']} (Mb)",
                                   fontsize=6.6, labelpad=3)
        fig.text(upper.x0, .968, "abc"[column], fontsize=8, fontweight="bold", va="top")
        fig.text(upper.x0 + .020, .968, f"{panel['title']} - chromosome {panel['chromosome']}",
                 fontsize=7, va="top")
    label_x = axes[0, 0].get_position().x0 - .043
    for row, text in ((0, "Effective\ncoefficient"), (1, "Locus-specific\nadjustment")):
        box = axes[row, 0].get_position()
        fig.text(label_x, (box.y0 + box.y1) / 2, text, ha="center", va="center", rotation=90, fontsize=6.2)
    top, bottom = axes[2, 0].get_position().y1, axes[3, 0].get_position().y0
    fig.text(label_x, (top + bottom) / 2, "GWAS association\n" + r"$-\log_{10}(P)$",
             ha="center", va="center", rotation=90, fontsize=6.2)
    sources = ["AFR", "EAS", "AMR", "CSA"]
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=source_colors[source],
                      markeredgecolor="none", markersize=3.3, label=source) for source in sources]
    center = (axes[3, 0].get_position().x0 + axes[3, 1].get_position().x1) / 2
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(center, .016), ncol=4,
               frameon=False, handletextpad=.25, columnspacing=.75, fontsize=6,
               title="Non-EUR ancestry", title_fontsize=6)
    return fig


# Read dedicated analysis inputs and export the combined map.
def main():
    args = parse_args(__doc__, interpretation=True, regional=True)
    panels = [load_regional_panel(args.data_root, spec, args.analysis_dir, args.gwas_template) for spec in regional_specs[:3]]
    with plt.rc_context(style):
        save_figure(build_figure(panels), args.output_dir / "Figure_5.pdf")


if __name__ == "__main__":
    main()
