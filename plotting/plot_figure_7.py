"""Sex-stratified prediction from pooled-sex training, with paired bootstrap tests."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .utils import load_sex_results, parse_args
from .plot_utils import draw_comparison, save_figure, style, style_axis


def build_figure(values, tests):
    methods = ["Female PRS", "Male PRS", "Nonwindowed MultiPRS", "BLocPRS mb1"]
    colors = ["#D9776A", "#72A58B", "#6BAED6", "#6A51A3"]
    values = values.loc[values.stratum.isin(("Female", "Male")) & values.method.isin(methods)]
    fig = plt.figure(figsize=(183/25.4, 120/25.4))
    grid = fig.add_gridspec(2, 4, left=.085, right=.985, bottom=.10, top=.81, wspace=.20, hspace=.46)
    transfer_shift = (grid[1, 1].get_position(fig).x0-grid[1, 0].get_position(fig).x0)/2
    offsets = dict(zip(methods, (-.30, -.10, .10, .30)))
    ymax = float((values.value+values.err).max())*1.35
    phenotypes = [("URT_SEX", "Urate"), ("LOG_CRTN_SEX", "Creatinine\u2020"),
                  ("WHR_SEX", "Waist-to-hip ratio"), ("LOG_TST_SEX", "Testosterone\u2020")]
    for column, (phenotype, title) in enumerate(phenotypes):
        for row, scenario in enumerate(("ukbb_native", "ukbb_to_cartagene")):
            if row == 1 and phenotype == "LOG_TST_SEX":
                continue
            position = grid[row, column].get_position(fig)
            ax = fig.add_axes([position.x0+(transfer_shift if row else 0), position.y0, position.width, position.height])
            ax.set_title(title, fontsize=7.5, pad=6)
            cell = values.loc[values.phenotype.eq(phenotype) & values.scenario.eq(scenario)]
            for method, color in zip(methods, colors):
                group = cell.loc[cell.method.eq(method)].set_index("stratum").loc[["Female", "Male"]]
                positions = np.arange(2)+offsets[method]
                ax.bar(positions, group.value, width=.18, color=color, edgecolor="#666666", lw=.4)
                ax.errorbar(positions, group.value, yerr=group.err, fmt="none", ecolor="#363636", elinewidth=.55)

            # Convert one typographic point to data units for consistent bracket spacing.
            unit = ymax/(ax.get_position().height*fig.get_figheight()*72)
            for position, stratum in enumerate(("Female", "Male")):
                group = cell.loc[cell.stratum.eq(stratum)].set_index("method")
                top = float((group.value+group.err).max())
                comparisons = tests.loc[tests.scenario.eq(scenario) & tests.phenotype.eq(phenotype)
                                        & tests.stratum.eq(stratum)].set_index("baseline")
                level = 0

                # Main Figure 7 compares the sex-matched source, not the best observed source.
                for baseline in ("Nonwindowed MultiPRS", stratum+" PRS"):
                    p = comparisons.loc[baseline, "p_wald_two_sided"]
                    if p < .05:
                        draw_comparison(ax, position+offsets[baseline], position+offsets["BLocPRS mb1"],
                                        p, top+(1+6.5*level)*unit, .8*unit)
                        level += 1
            ax.set(xticks=[0, 1], xticklabels=["Female", "Male"], xlim=(-.55, 1.55), ylim=(0, ymax))
            ax.tick_params(labelsize=7, labelleft=column == 0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            style_axis(ax)
            if column == 0:
                ax.set_ylabel(r"Incremental $R^2$", fontsize=7)
    fig.legend([Patch(facecolor=c, edgecolor="#666666", lw=.4) for c in colors],
               ["Female PRS", "Male PRS", "MultiPRS", "BLocPRS"], loc="upper center",
               bbox_to_anchor=(.535, .935), ncol=4, frameon=False, fontsize=7, handlelength=1.2)
    fig.text(.535, .875, "Held-out UK Biobank", ha="center", fontsize=7.5)
    fig.text(.535, .445, r"UK Biobank $\rightarrow$ CARTaGENE transfer", ha="center", fontsize=7.5)
    fig.text(.065, .885, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(.065, .455, "b", fontsize=8, fontweight="bold", va="top")
    fig.text(.535, .025, "Evaluation group", ha="center", fontsize=7)
    return fig


def main():
    args = parse_args(__doc__)
    values, tests = load_sex_results(args.results)
    with plt.rc_context(style):
        save_figure(build_figure(values, tests), args.output_dir / "Figure_7.pdf")


if __name__ == "__main__":
    main()
