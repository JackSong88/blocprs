"""Structured source controls and expanded-covariate adjustment with paired bootstrap SEs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ..utils import (attach_uncertainty, metric_labels, metric_tokens, one_row, parse_args,
                     phenotype_label, read_comparisons, read_results, select_results)
from ..plot_utils import save_figure, style, style_axis


targets = (("LDL_MA", "continuous"), ("URT_SEX", "continuous"), ("LOG_CRTN_SEX", "continuous"),
           ("ASTHMA_MA", "binary"), ("CAD_MT", "binary"))
panels = ("all_real", "all_real_plus_controls", "controls_only")
colors = ("#6A51A3", "#3182BD", "#A6A6A6")
windows = ("mb1", "mb3", "mb5")


# All panels share evaluation participants; expanded designs alter covariates, not scores.
def load_results(path):
    data = read_results(path)
    comparisons = read_comparisons(path)
    records, differences = [], []
    for phenotype, family in targets:
        frame = select_results(data, phenotype=phenotype, cohort="ukbb", group="all",
                               train_group="all", train_size="full", metric=metric_tokens[family])
        counts = []
        for mb in (1, 3, 5):
            window = f"mb{mb}"
            for panel in panels:
                row = one_row(frame, method="blocprs", window_mb=mb, panel=panel, covariate_set="standard")
                counts.append(row.n)
                records.append(dict(row, phenotype=phenotype, family=family, window=window))

            for design, label in (("standard", "Standard"), ("expanded", "Expanded")):
                baseline = one_row(frame, method="multiprs", window_mb=0,
                                   panel="all_real", covariate_set=design)
                target = one_row(frame, method="blocprs", window_mb=mb,
                                 panel="all_real", covariate_set=design)
                counts += [baseline.n, target.n]

                # Paired uncertainty includes covariance between the two estimates.
                comparison = one_row(comparisons, phenotype=phenotype, cohort="ukbb", group="all",
                                     train_group="all", panel="all_real", covariate_set=design,
                                     metric=metric_tokens[family], method="blocprs", window_mb=mb,
                                     reference="multiprs", reference_window_mb=0)
                difference = target.value - baseline.value

                if comparison.n != target.n or not np.isclose(comparison.difference, difference, atol=1e-10, rtol=1e-7):
                    raise ValueError("Expanded-covariate paired difference does not match its point estimates.")

                differences.append(dict(phenotype=phenotype, family=family, window=window, design=label,
                                        value=difference, bootstrap_se=comparison.difference_se, n=target.n))

        if len(set(counts)) != 1:
            raise ValueError(f"Structured-control evaluations use different sample counts: {phenotype}")

    return attach_uncertainty(pd.DataFrame(records)), pd.DataFrame(differences)


# Set limits by outcome family, retaining full signed uncertainty intervals.
def panel_limits(frame, minimum_high=0):
    low = min(0, (frame.value-frame.bootstrap_se).min())
    high = max(minimum_high, (frame.value+frame.bootstrap_se).max())
    return low*1.1, high*1.15


# Draw source-control performance alongside paired BLocPRS-minus-MultiPRS gains.
def build_figure(data, expanded):
    width, height = 183, 216
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    fig.text(.5, 212/height, "Held-out UK Biobank", ha="center", va="center", fontsize=8.5)
    for letter, title, left in (("a", "Out-of-phenotype source controls", 19),
                                ("b", "Expanded-covariate adjustment", 108)):
        fig.text(left/width, 203/height, letter, fontsize=8, fontweight="bold", va="center")
        fig.text((left+7)/width, 203/height, title, fontsize=7.5, va="center")
    for row, (phenotype, family) in enumerate(targets):
        left_ax = fig.add_axes([19/width, (157-35*row)/height, 68/width, 24/height])
        right_ax = fig.add_axes([113/width, (157-35*row)/height, 60/width, 24/height])
        for index, (panel, color) in enumerate(zip(panels, colors)):
            frame = data.loc[data.phenotype.eq(phenotype) & data.panel.eq(panel)].set_index("window").loc[list(windows)]
            left_ax.bar(np.arange(3)*.8+(index-1)*.17, frame.value, width=.15, color=color,
                        edgecolor="none", yerr=frame.bootstrap_se,
                        error_kw=dict(ecolor="#414141", elinewidth=.48, capsize=0))
        for index, (design, color) in enumerate((("Standard", "#3182BD"), ("Expanded", "#6A51A3"))):
            frame = expanded.loc[expanded.phenotype.eq(phenotype) & expanded.design.eq(design)].set_index("window").loc[list(windows)]
            right_ax.bar(np.arange(3)*.8+(index-.5)*.17, frame.value, width=.15, color=color,
                         edgecolor="none", yerr=frame.bootstrap_se,
                         error_kw=dict(ecolor="#414141", elinewidth=.48, capsize=0))
        left_ax.set_ylim(panel_limits(data.loc[data.family.eq(family)]))
        right_ax.set_ylim(panel_limits(expanded.loc[expanded.family.eq(family)], minimum_high=.001))
        left_ax.set_ylabel(metric_labels[family], fontsize=7.5, labelpad=4)
        right_ax.set_ylabel(r"$R^2_{\mathrm{BLocPRS}}-R^2_{\mathrm{MultiPRS}}$", fontsize=7.5, labelpad=4)
        title = {"URT_SEX": "Urate", "LOG_CRTN_SEX": "Creatinine\u2020"}.get(phenotype, phenotype_label(phenotype))
        for ax in (left_ax, right_ax):
            style_axis(ax)
            ax.set_xlim(-.35, 1.95)
            ax.set_xticks(np.arange(3)*.8, ["1 Mb", "3 Mb", "5 Mb"])
            ticks = MaxNLocator(nbins=3).tick_values(*ax.get_ylim())
            ax.set_yticks([tick for tick in ticks if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]])
            ax.spines["bottom"].set_position(("data", 0))
            ax.tick_params(labelsize=6.5, length=2.2, width=.5)
            ax.set_title(title, fontsize=8, pad=5)
            if row == 4:
                ax.set_xlabel("Window size", fontsize=7.5, labelpad=4)
    fig.legend([Patch(facecolor=color, edgecolor="none") for color in colors],
               ["Full panel", "Full + controls", "Controls only"], loc="lower center",
               bbox_to_anchor=(53/width, 191/height), ncol=3, fontsize=6.5, frameon=False,
               handlelength=1.3, columnspacing=1)
    fig.legend([Patch(facecolor=color, edgecolor="none") for color in ("#3182BD", "#6A51A3")],
               ["Standard adjustment", "Expanded adjustment"], loc="lower center",
               bbox_to_anchor=(143/width, 191/height), ncol=2, fontsize=6.5, frameon=False,
               handlelength=1.3, columnspacing=1)
    return fig


# Plot saved estimates and paired bootstrap uncertainty.
def main():
    args = parse_args(__doc__, supplementary=True, collection="structured_controls")
    data, expanded = load_results(args.results)
    with plt.rc_context(style):
        save_figure(build_figure(data, expanded), args.output_dir / "suppl_structured_controls.pdf")


if __name__ == "__main__":
    main()
