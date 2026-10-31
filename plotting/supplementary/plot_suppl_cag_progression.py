"""Transfer model progression with separately selected, optimistic CaG single-source comparators."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..utils import load_progression, parse_args
from ..plot_utils import grouped_estimates, save_figure, style


methods = ("Best single-source PRS", "Best regional single-source PRS", "MultiPRS",
           "Fixed-composition regional MultiPRS", "BLocPRS (1 Mb)")
labels = ("Best single-source PRS", "Best single-source BLocPRS", "MultiPRS",
          "BLocPRS (fixed composition)", "BLocPRS")
colors = ("#A6A6A6", "#555555", "#6BAED6", "#3182BD", "#6A51A3")
markers = ("o", "o", "s", "^", "D")
# The shared loader checks complete source panels, matched counts, splits and saved SEs.
# CaG source maxima remain optimistic comparators, as in this supplementary analysis.
def load_results(path):
    data = load_progression(path, "cartagene")
    labels = dict(zip(["Best single-source PRS", "Best regional single-source PRS",
                       "Nonwindowed MultiPRS", "Fixed-composition regional MultiPRS",
                       "BLocPRS mb1"], methods))
    data["method"] = data.method.map(labels)
    return data


# Use the same offset markers and absolute metric scales as native model progression.
def build_figure(data):
    fig = grouped_estimates(data, methods, colors, "UK Biobank to CARTaGENE transfer",
                             order_by=methods[-1], markers=markers, open_first=True, sex_labels=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.07, top=.88, wspace=.32)
    fig._suptitle.set_y(.95)
    for legend in list(fig.legends):
        legend.remove()
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=3.8,
                      markerfacecolor="white" if index == 0 else color,
                      markeredgecolor=color, markeredgewidth=.55)
               for index, (color, marker) in enumerate(zip(colors, markers))]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995), borderaxespad=0,
               ncol=len(methods), frameon=False, fontsize=7, handlelength=.85, handletextpad=.4, columnspacing=.65)
    return fig


# Export the transfer comparison without fitting models or recomputing uncertainty.
def main():
    args = parse_args(__doc__, supplementary=True, collection="progression")
    data = load_results(args.results)
    with plt.rc_context(style):
        save_figure(build_figure(data), args.output_dir / "suppl_cag_progression.pdf")


if __name__ == "__main__":
    main()
