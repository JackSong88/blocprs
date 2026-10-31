"""Signed source contributions in the same top-1% windows as the composition figures."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..utils import load_regional_interpretation, parse_args, phenotype_label, regional_groups
from ..plot_utils import contribution_palette, draw_signed, ordered_sources, save_figure, style


def build_figure(values, batch, title, colors):
    pitch = 41 if all(p.endswith("_MT") for p in batch) else 36
    height = ((len(batch)+1)//2)*pitch+30
    fig = plt.figure(figsize=(183/25.4, height/25.4))
    fig.suptitle(f"Top 1% reference-ranked windows - {title}", fontsize=7, y=1-2/height)
    for index, phenotype in enumerate(batch):
        row, block = divmod(index, 2)
        if len(batch) % 2 and index == len(batch)-1:
            block = .5
        left, top = 14+86*block, 23+pitch*row
        rows = values.loc[values.phenotype.eq(phenotype)]
        sources = ordered_sources(rows.expert_label)
        fig.text((left+36.5)/183, 1-(top-3.5)/height, phenotype_label(phenotype).replace("\n", " "),
                 ha="center", fontsize=7)
        for column, (component, heading) in enumerate((("reference", "Genome-wide reference"), ("effective", "Full BLocPRS"))):
            ax = fig.add_axes([(left+40*column)/183, 1-(top+22)/height, 33/183, 22/height])

            # Reset the seed to reuse the same horizontal jitter sequence in both panels.
            draw_signed(ax, rows, component, np.random.default_rng(20260817), colors)
            if row == 0:
                fig.text((left+40*column+16.5)/183, 1-11/height, heading, ha="center", fontsize=7)
            ax.tick_params(axis="y", labelleft=column == 0)
            if phenotype.endswith("_MT"):
                ax.set_xticklabels(sources, rotation=40, ha="right", rotation_mode="anchor", fontsize=7)
    fig.supylabel("Signed prediction-scale contribution", fontsize=7, x=.008)
    fig.supxlabel("PRS source", fontsize=7, y=9/height)
    fig.legend([Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor="#333333",
                       ms=3.3, markeredgewidth=.5)], ["Unweighted window mean"], loc="lower center",
               bbox_to_anchor=(.5, 0), fontsize=7, frameon=False)
    return fig


def main():
    args = parse_args(__doc__, supplementary=True, interpretation=True)
    _, ratios, top = load_regional_interpretation(args.data_root, args.analysis_dir)
    colors = contribution_palette(top.expert_label)
    with plt.rc_context(style):
        for name, title, batch in regional_groups(ratios.phenotype.drop_duplicates()):
            figure = build_figure(top.loc[top.phenotype.isin(batch)], batch, title, colors)
            save_figure(figure, args.output_dir / f"suppl_top1_signed_contributions_{name}.pdf")


if __name__ == "__main__":
    main()
