"""Training-scale contribution profiles, magnitude ratios, and source-share changes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..utils import (load_regional_interpretation, parse_args, phenotype_label,
                     regional_groups, target_source)
from ..plot_utils import (contribution_palette, ordered_sources, reference_bins,
                          save_figure, style, style_axis)


def trim_ticks(ax, count=3):
    ticks = MaxNLocator(nbins=count).tick_values(*ax.get_ylim())
    ax.set_yticks([tick for tick in ticks if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]])


def profile_figures(means, ratios):
    colors = contribution_palette(means.expert_label.unique())
    for name, title, batch in regional_groups(ratios.phenotype.drop_duplicates()):
        nrows = (len(batch)+1)//2
        height = nrows*35+34
        fig = plt.figure(figsize=(183/25.4, height/25.4))
        fig.suptitle(title, fontsize=7, y=1-3/height)
        all_sources = ordered_sources(means.loc[means.phenotype.isin(batch)].expert_label.unique())
        for index, phenotype in enumerate(batch):
            row, block = divmod(index, 2)
            selected = means.loc[means.phenotype.eq(phenotype)]
            sources = ordered_sources(selected.expert_label.unique())
            target = target_source(phenotype, sources) if name == "related_traits" else None
            related = [source for source in sources if source != target]
            source_colors = colors if target is None else {
                target: "#333333", **dict(zip(related, ["#0072B2", "#D55E00", "#009E73", "#AA6BA5"]))}
            if len(batch) % 2 and index == len(batch)-1:
                block = .5
            left = 12+88.5*block
            top = 1-(20+row*35)/height
            show_percentiles = row == nrows-1 or (len(batch) % 2 and row == nrows-2)
            empty_bins = [i for i, label in enumerate(reference_bins) if not selected.bin.astype(str).eq(label).any()]
            low, high = selected[["reference", "adjustment", "effective"]].to_numpy().min(), selected[["reference", "adjustment", "effective"]].to_numpy().max()
            span = max(high-low, .0001)
            for column, (component, heading) in enumerate((("effective", "Full BLocPRS"),
                    ("reference", "Genome-wide\nreference"), ("adjustment", "Locus-specific\nadjustment"))):
                ax = fig.add_axes([(left+27*column)/183, top-22/height, 25/183, 22/height])
                for source in sources:
                    values = selected.loc[selected.expert_label.eq(source)].set_index("bin").reindex(reference_bins)[component]
                    ax.plot(range(7), values, color=source_colors[source], marker="o", ms=1.7,
                            lw=.9 if source == target else .65, zorder=4 if source == target else 3)

                    # Dashed segments bridge empty rank bins without inventing observations.
                    observed = np.flatnonzero(np.isfinite(values.to_numpy()))
                    for a, b in zip(observed[:-1], observed[1:]):
                        if b-a > 1:
                            if not set(range(a+1, b)).issubset(empty_bins):
                                raise ValueError("A nonempty rank bin has missing contribution values")
                            ax.plot([a, b], values.iloc[[a, b]], color=source_colors[source], lw=.55, ls="--", zorder=2)
                style_axis(ax)
                ax.axhline(0, color="#999999", lw=.4, ls=":", zorder=0)
                ax.set_ylim(min(0, low)-.07*span, high+.10*span)
                ax.set_xticks(range(7), reference_bins, rotation=60, ha="right", rotation_mode="anchor")
                trim_ticks(ax, 2)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.tick_params(axis="x", labelbottom=show_percentiles, labelsize=7, pad=2)
                ax.tick_params(axis="y", labelleft=column == 0, labelsize=5.9)
                if row == 0:
                    ax.set_title(heading, fontsize=6.5, pad=14)
                if column == 0:
                    ax.text(0, 1.06, phenotype_label(phenotype).replace("\n", " "), transform=ax.transAxes, fontsize=6.6)
                if name == "related_traits" and column == 2:
                    ax.legend([Line2D([], [], color=source_colors[source], lw=.8) for source in related], related,
                              title="Related-trait PRSs", loc="upper left", bbox_to_anchor=(.02, .98),
                              borderaxespad=0, borderpad=.1, frameon=False, fontsize=5.7, title_fontsize=5.7,
                              handlelength=.9, handletextpad=.35, labelspacing=.15)
        if name == "related_traits":
            fig.legend([Line2D([], [], color="#333333", lw=.9, marker="o", ms=2)], ["Target-phenotype PRS"],
                       bbox_to_anchor=(.53, 2/height), loc="lower center", frameon=False, fontsize=6, borderaxespad=0)
        else:
            fig.legend([Line2D([], [], color=colors[source], lw=.8, marker="o", ms=1.8) for source in all_sources],
                       all_sources, bbox_to_anchor=(.53, 2/height), loc="lower center", ncol=len(all_sources),
                       frameon=False, fontsize=5.5, handlelength=1, columnspacing=.9, borderaxespad=0)
        fig.supylabel("Signed prediction-scale contribution", fontsize=7, x=.012)
        fig.supxlabel("Genome-wide reference contribution (window percentile)", fontsize=7, x=.53, y=14/height)
        yield f"suppl_regional_contributions_{name}.pdf", fig


def magnitude_figure(ratios):
    order = ratios.phenotype.drop_duplicates().tolist()
    order = [p for p in order if "_ADJ_" not in p] + [p for p in order if "_ADJ_" in p]
    matrix = ratios.pivot(index="phenotype", columns="bin", values="ratio").loc[order, reference_bins]
    fig, (top, ax) = plt.subplots(2, 1, figsize=(183/25.4, 150/25.4), gridspec_kw={"height_ratios": [1, 4]})
    fig.subplots_adjust(left=.11, right=.91, bottom=.09, top=.92, hspace=.18)
    core = matrix.loc[[p for p in order if "_ADJ_" not in p]]
    top.plot(range(7), core.mean(), "o-", color="#6A51A3", ms=3, lw=1, label="Core 25 - mean")
    if len(matrix) > len(core):
        top.plot(range(7), matrix.mean(), "s--", color="#666666", ms=2.5, lw=.8,
                 label=f"All {len(matrix)} - mean")
    style_axis(top)
    top.set_xticks(range(7), reference_bins, fontsize=6)
    top.set_ylim(0, max(core.mean().max(), matrix.mean().max())*1.25)
    trim_ticks(top)
    top.set_ylabel("Adjustment contribution\n(% of reference\ncontribution)", fontsize=6.5)
    top.legend(fontsize=6.2, frameon=False, loc="upper left")

    # Undefined ratios remain blank and are excluded from phenotype means.
    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad("white")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0)
    ax.set_yticks(range(len(order)), [phenotype_label(p).replace("\n", " ") for p in order], fontsize=6)
    ax.set_xticks(range(7), reference_bins, rotation=35, ha="right", fontsize=6)
    style_axis(ax)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Genome-wide reference contribution (window percentile)", fontsize=6.5)
    box = ax.get_position()
    colorbar = fig.colorbar(image, cax=fig.add_axes([.935, box.y0, .014, box.height]))
    colorbar.ax.tick_params(labelsize=6, length=2, width=.5)
    colorbar.set_label("Adjustment contribution (% of reference contribution)", fontsize=6.5)
    return fig


def composition_figures(top):
    rng = np.random.default_rng(20260817)
    colors = contribution_palette(top.expert_label.unique())
    groups = {name: batch for name, _, batch in regional_groups(top.phenotype.drop_duplicates())}
    for name in ("multi_ancestry_sex_specific", "related_traits"):
        combined = name == "multi_ancestry_sex_specific"
        sex = groups["sex_specific"] if combined else []
        primary = groups["multi_ancestry"] if combined else groups["related_traits"]
        batch = primary+sex
        limit = max(10, np.ceil(top.loc[top.phenotype.isin(batch), "shift"].abs().max()/10)*10)
        row_sizes = []
        for group in (primary, sex):
            if group:
                rows = (len(group)+3)//4
                size, extra = divmod(len(group), rows)
                row_sizes.extend([size+(row < extra) for row in range(rows)])
        primary_rows = (len(primary)+3)//4
        positions = [(row, col) for row, count in enumerate(row_sizes) for col in range(count)]
        plot_height = 32 if combined else 38
        pitch = plot_height+(13 if combined else 18)
        gap, plot_top = (6 if sex else 0), (20 if combined else 14)
        height = len(row_sizes)*pitch+17+gap+(plot_top-18)
        fig = plt.figure(figsize=(183/25.4, height/25.4))
        title = "Composition shifts in the top 1% of windows" + ("" if combined else " - Related traits")
        fig.suptitle(title, fontsize=8, y=1-4/height)
        if sex:
            for letter, heading, mm in (("a", "Multi-ancestry", plot_top-9),
                    ("b", "Sex-specific", plot_top+primary_rows*pitch+gap-8)):
                fig.text(7/183, 1-mm/height, letter, fontsize=8, fontweight="bold")
                fig.text(13/183, 1-mm/height, heading, fontsize=7, color="#444444")
        for index, phenotype in enumerate(batch):
            row, col = positions[index]
            step = 171/row_sizes[row]
            offset = gap if phenotype in sex else 0
            ax = fig.add_axes([(13+col*step)/183, 1-(plot_top+row*pitch+offset+plot_height)/height,
                               (step-5)/183, plot_height/height])
            selected = top.loc[top.phenotype.eq(phenotype)]
            sources = ordered_sources(selected.expert_label)
            for position, source in enumerate(sources):
                group = selected.loc[selected.expert_label.eq(source)]
                values, color = group["shift"].to_numpy(), colors[source]
                box = ax.boxplot([values], positions=[position], widths=.48, whis=(5, 95), patch_artist=True,
                                 showfliers=False, medianprops={"color": "#5A5A5A", "linewidth": .92},
                                 boxprops={"facecolor": color, "edgecolor": color, "linewidth": .68},
                                 whiskerprops={"color": color, "linewidth": .68}, capprops={"color": color, "linewidth": .68})
                box["boxes"][0].set_alpha(.16)
                ax.scatter(position+rng.uniform(-.14, .14, len(values)), values, s=4.8,
                           color=color, edgecolors="none", alpha=.38, rasterized=True, zorder=3)

                # The diamond weights each window by its total absolute reference contribution.
                ax.scatter([position], [np.average(values, weights=group.G)], marker="D", s=11.5,
                           facecolor="white", edgecolor="#333333", linewidth=.48, zorder=5)
            style_axis(ax)
            ax.axhline(0, color="#999999", ls=":", lw=.5)
            ax.set(ylim=(-limit, limit), xlim=(-.6, len(sources)-.4))
            rotate = name == "related_traits" or len(sources) > 5
            ax.set_xticks(range(len(sources)), sources, rotation=45 if rotate else 0,
                          ha="right" if rotate else "center", rotation_mode="anchor", fontsize=6)
            trim_ticks(ax, 2)
            ax.tick_params(axis="y", labelleft=col == 0, labelsize=6)
            ax.set_title(phenotype_label(phenotype).replace("\n", " "), fontsize=6.8, pad=3)
        fig.supylabel("Change in absolute contribution share (percentage points)", fontsize=7, x=.012)
        fig.legend([Line2D([], [], marker="D", color="#333333", markerfacecolor="white", markeredgewidth=.55,
                           ls="none", ms=3)], ["Reference-contribution-weighted mean"], loc="lower center",
                   bbox_to_anchor=(.5, 1/height), frameon=False, fontsize=6)
        yield f"suppl_source_composition_{name}.pdf", fig


def main():
    args = parse_args(__doc__, supplementary=True, interpretation=True)
    means, ratios, top = load_regional_interpretation(args.data_root, args.analysis_dir)
    with plt.rc_context(style):
        for name, figure in profile_figures(means, ratios):
            save_figure(figure, args.output_dir / name)
        save_figure(magnitude_figure(ratios), args.output_dir / "suppl_regional_contribution_magnitude.pdf")
        for name, figure in composition_figures(top):
            save_figure(figure, args.output_dir / name)


if __name__ == "__main__":
    main()
