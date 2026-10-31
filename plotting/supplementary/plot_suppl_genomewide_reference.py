"""Compare MultiPRS with the fitted BLocPRS genome-wide component."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter

from ..plot_utils import save_figure, style, style_axis
from ..utils import core_phenotypes, metric_labels, parse_args, phenotype_label


methods = ["MultiPRS", "BLocPRS genome-wide component"]
colors = ["#6BAED6", "#6A51A3"]
scenarios = ["ukbb_native", "full_cag_transfer"]
missing_cag = {"LOG_TST_SEX", "DEM_MT"}


def load_results(data_root, analysis_dir=None):
    tables = []
    for phenotype, family in core_phenotypes:
        for scenario in scenarios:
            if scenario == scenarios[1] and phenotype in missing_cag:
                continue
            path = analysis_dir / phenotype / "ablation" / scenario / "reference.tsv"
            frame = pd.read_csv(path, sep="\t")

            if (len(frame) != 2 or set(frame.method) != set(methods)
                    or not frame.phenotype.eq(phenotype).all() or not frame.family.eq(family).all()
                    or not frame.scenario.eq(scenario).all() or not frame.window_size.eq("mb1").all()):
                raise ValueError(f"Incomplete or mismatched genome-wide-component comparison: {path}")

            values = frame[["value", "n", "bootstrap_se"]].to_numpy(dtype=float)
            if (not np.isfinite(values).all() or frame.n.nunique() != 1
                    or (frame.n <= 0).any() or frame.bootstrap_se.lt(0).any()):
                raise ValueError(f"Matched estimates and bootstrap SEs are required: {path}")

            tables.append(frame)
    return pd.concat(tables, ignore_index=True)



def build_figure(data):
    width, height = 183, 185
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    for col, title in enumerate(["Held-out UK Biobank", r"UK Biobank $\rightarrow$ CARTaGENE transfer"]):
        fig.text((25+88*col+32)/width, 177/height, title, ha="center", va="center", fontsize=7.5)
    for row, family in enumerate(["continuous", "binary"]):
        selected_family = data.loc[data.family.eq(family)]

        # Both cohorts retain the native-UKB trait order and a common horizontal scale.
        baseline = selected_family.loc[selected_family.scenario.eq(scenarios[0]) & selected_family.method.eq(methods[0])]
        order = baseline.sort_values(["value", "phenotype"], kind="stable").phenotype.tolist()
        lower = min(0, float((selected_family.value-selected_family.bootstrap_se).min()))*1.07
        upper = float((selected_family.value+selected_family.bootstrap_se).max())*1.07
        ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper)
        ticks = ticks[(ticks >= lower) & (ticks <= upper)]
        bottom, panel_height = (105, 63) if row == 0 else (24, 60)
        for col, scenario in enumerate(scenarios):
            ax = fig.add_axes([(25+88*col)/width, bottom/height, 64/width, panel_height/height])
            selected = selected_family.loc[selected_family.scenario.eq(scenario)]
            for index in range(0, len(order), 2):
                ax.axhspan(index-.49, index+.49, color="#F2F3F4", linewidth=0, zorder=0)
            for j, (method, color, marker) in enumerate(zip(methods, colors, ["s", "o"])):
                values = selected.loc[selected.method.eq(method)].set_index("phenotype").reindex(order)
                ax.errorbar(values.value, np.arange(len(order))+(j-.5)*.30, xerr=values.bootstrap_se,
                            fmt=marker, linestyle="none", markersize=2.8, markerfacecolor=color,
                            markeredgecolor=color, markeredgewidth=.5, ecolor=color,
                            elinewidth=.55, capsize=0, zorder=3)
            for index, phenotype in enumerate(order):
                if phenotype not in set(selected.phenotype):
                    ax.text(upper*.045, index, "-", fontsize=9, va="center", color="#777777")
            ax.set(ylim=(len(order)-.5, -.5), xlim=(lower, upper), xticks=ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:.2f}"))
            ax.set_yticks(range(len(order)), [phenotype_label(p) for p in order], fontsize=6)
            style_axis(ax)
            ax.tick_params(labelsize=6, length=2.3, width=.5)
            ax.set_xlabel(metric_labels[family], fontsize=7, labelpad=4)
            ax.text(0, 1.045, chr(65+2*row+col), transform=ax.transAxes, fontsize=8.5,
                    fontweight="bold", va="bottom")
            ax.text(.5, 1.045, family.capitalize(), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7)
    handles = [Line2D([], [], linestyle="none", marker=marker, color=color,
                      markerfacecolor=color, markeredgewidth=.5, markersize=3.5)
               for color, marker in zip(colors, ["s", "o"])]
    fig.legend(handles, methods, loc="lower center", bbox_to_anchor=(.5, .025), ncol=2,
               fontsize=7, frameon=False, handlelength=1, columnspacing=2.4)
    return fig


def main():
    args = parse_args(__doc__, supplementary=True, interpretation=True)
    data = load_results(args.data_root, args.analysis_dir)
    with plt.rc_context(style):
        save_figure(build_figure(data), args.output_dir / "suppl_genomewide_reference.pdf")


if __name__ == "__main__":
    main()
