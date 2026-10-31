"""Core-phenotype MTL-PRS comparison with matched evaluation and bootstrap uncertainty."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..utils import core_phenotypes, metric_labels, phenotype_label
from ..plot_utils import save_figure, style, style_axis


methods = ("MTL-PRS", "Fixed-composition regional MultiPRS", "BLocPRS (1 Mb)")
labels = ("MTL-PRS", "BLocPRS (fixed composition)", "BLocPRS (1 Mb)")
colors = ("#D55E00", "#3182BD", "#6A51A3")
markers = ("o", "^", "D")


# Supply the per-phenotype results from the packaged MTL-PRS collection stage.
def parse_args(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "figures/supplementary")
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="Result tables covering the 25 core analyses and three methods, with bootstrap SEs.")
    return parser.parse_args(argv)


# Require one matched estimate and SE for every core phenotype and method.
def load_result_table(paths):
    data = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    data = data[["phenotype", "family", "method", "value", "n", "bootstrap_se"]].copy()
    keys = pd.MultiIndex.from_frame(data[["phenotype", "family", "method"]])
    expected = {(phenotype, family, method) for phenotype, family in core_phenotypes for method in methods}

    if keys.has_duplicates or set(keys) != expected:
        raise ValueError("MTL-PRS results must contain each of the 25 core phenotypes and three methods exactly once.")

    values = data[["value", "n", "bootstrap_se"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(values[:, 1] <= 0)
        or np.any(values[:, 1] != np.floor(values[:, 1])) or np.any(values[:, 2] < 0)):
        raise ValueError("Estimates and SEs must be finite, SEs nonnegative, and sample counts positive integers.")

    if data.groupby("phenotype").n.nunique().ne(1).any():
        raise ValueError("The three methods must use the same evaluation sample count per phenotype.")

    order = pd.MultiIndex.from_tuples([(phenotype, method) for phenotype, _ in core_phenotypes for method in methods])
    return data.set_index(["phenotype", "method"]).loc[order].reset_index()


# Keep the phenotype order and method offsets fixed across the two outcome families.
def build_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 125 / 25.4))
    fig.subplots_adjust(left=.10, right=.985, bottom=.14, top=.89, wspace=.32)
    fig.suptitle("Held-out UK Biobank", fontsize=8, y=.97)
    for index, (ax, family) in enumerate(zip(axes, ("continuous", "binary"))):
        frame = data.loc[data.family.eq(family)]
        order = frame.phenotype.drop_duplicates().tolist()
        y = np.arange(len(order))
        for j, (method, color, marker) in enumerate(zip(methods, colors, markers)):
            group = frame.loc[frame.method.eq(method)].set_index("phenotype").loc[order]
            ax.errorbar(group.value, y+(j-1)*.215, xerr=group.bootstrap_se, fmt=marker,
                        linestyle="none", markersize=2.5 if j == 2 else 3, markerfacecolor=color,
                        markeredgecolor=color, markeredgewidth=.55, ecolor=color, elinewidth=.55, capsize=0)
        style_axis(ax)
        ax.set_yticks(y, [phenotype_label(phenotype) for phenotype in order], fontsize=6.4)
        ax.invert_yaxis()
        low = min(0, (frame.value-frame.bootstrap_se).min())
        high = (frame.value+frame.bootstrap_se).max()
        ax.set_xlim(low*1.06, high*1.06)
        ticks = MaxNLocator(nbins=4).tick_values(*ax.get_xlim())
        ax.set_xticks([tick for tick in ticks if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]])
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel(metric_labels[family], fontsize=7)
        ax.set_title(family.capitalize(), fontsize=7, pad=6)
        ax.text(0, 1.055, "ab"[index], transform=ax.transAxes, fontsize=8, fontweight="bold")
        ax.set_axisbelow(True)
        for row in range(0, len(order), 2):
            ax.axhspan(row-.48, row+.48, color="#F6F7F7", linewidth=0, zorder=0)
    center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=3.8,
                      markerfacecolor=color, markeredgecolor=color, markeredgewidth=.55)
               for color, marker in zip(colors, markers)]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(center, .027), borderaxespad=0,
               ncol=3, frameon=False, fontsize=7, handlelength=1, columnspacing=1.4)
    return fig


# Export the comparison using the standard errors supplied with its results.
def main():
    args = parse_args()
    data = load_result_table(args.results)
    with plt.rc_context(style):
        save_figure(build_figure(data), args.output_dir / "suppl_mtlprs.pdf")


if __name__ == "__main__":
    main()
