"""Plot LDL prediction from no-UKB VIPRS-GS, VIPRS-EM, and SSLPRS-GS experts."""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from ..plot_utils import compact_colors, draw_comparison, save_figure, style, style_axis
from ..utils import attach_uncertainty, method_row, paired_tests, parse_args, read_results, select_results


experts = [("VIPRSGS", "VIPRS-GS"), ("VIPRSEM", "VIPRS-EM"), ("SSLPRSGS", "SSLPRS-GS")]
groups = [("All", "All", "Full"), ("Coarse Ancestry", "EUR", "EUR"),
          ("Coarse Ancestry", "non-EUR", "Non-EUR")]
methods = ["Best single-source PRS", "MultiPRS", "BLocPRS (1 Mb)"]
cohorts = ["ukbb", "cartagene"]
keys = ["expert", "cohort", "stratum", "method"]


# Each expert panel has its own source inputs, but every comparison uses matched rows.
def load_results(path):
    data = read_results(path)
    records = []
    for expert, _ in experts:
        for cohort in cohorts:
            for group, label in (("all", "Full"), ("EUR", "EUR"), ("non-EUR", "Non-EUR")):
                frame = select_results(data, phenotype="LDL_MA", panel=expert, cohort=cohort,
                                       group=group, train_group="all", train_size="full",
                                       covariate_set="standard", metric="incremental_r2")
                components = {f"LDL_NOUKB_{expert}_{ancestry}" for ancestry in ("AFR", "CSA", "EUR")}

                if set(frame.loc[frame.method.eq("single"), "source"]) != components:
                    raise ValueError(f"Incomplete LDL expert panel: {expert}/{cohort}/{group}")

                for method, mb, display in zip(("single", "multiprs", "blocprs"), (0, 0, 1), methods):
                    row = method_row(frame, method, mb).to_dict()
                    row.update(method_token=method, method=display, expert=expert, stratum=label)
                    records.append(row)

    result = attach_uncertainty(pd.DataFrame(records))

    if result.groupby(["cohort", "stratum"]).n.nunique().gt(1).any():
        raise ValueError("LDL expert methods evaluate different participant counts.")

    tests = paired_tests(result, path, target=methods[-1], baselines=methods[:2])
    tests["expert"] = tests.panel
    return result, tests


def build_figure(data, comparisons):
    figure, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 112 / 25.4), sharey=True)
    figure.subplots_adjust(left=0.082, right=0.987, top=0.845, bottom=0.17, wspace=0.16, hspace=0.60)
    upper = ceil(((data.value + data.bootstrap_se).max() + 0.04) / 0.02) * 0.02
    lookup = data.set_index(keys).sort_index()
    for column, (expert, title) in enumerate(experts):
        box = axes[0, column].get_position()
        figure.text(box.x0, 0.957, "ABC"[column], fontsize=8.5, fontweight="bold", va="top")
        figure.text((box.x0 + box.x1) / 2, 0.957, title, fontsize=7.5, ha="center", va="top")
        for row, cohort in enumerate(cohorts):
            ax = axes[row, column]
            style_axis(ax)
            ax.set(ylim=(0, upper), xlim=(-0.47, 2.47), xticks=range(3), xticklabels=[g[2] for g in groups])
            ax.set_yticks(np.arange(0, upper + 0.001, 0.04))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(length=2.4, labelsize=6.3, left=column == 0, labelleft=column == 0)
            ax.spines["left"].set_visible(column == 0)
            for group_index, (_, _, group) in enumerate(groups):
                prefix = (expert, cohort, group)
                for method_index, (method, color) in enumerate(zip(methods, compact_colors[:3])):
                    record = lookup.loc[prefix + (method,)]
                    x = group_index + (method_index - 1) * 0.19
                    ax.bar(x, record.value, width=0.16, color=color, edgecolor="#666666", linewidth=0.22, zorder=3)
                    ax.errorbar(x, record.value, yerr=record.bootstrap_se, fmt="none",
                                ecolor="#333333", elinewidth=0.55, capsize=0, zorder=4)
                selected = lookup.loc[prefix]
                top = (selected.value + selected.bootstrap_se).max()
                tests = comparisons.loc[comparisons.expert.eq(expert) & comparisons.cohort.eq(cohort)
                                        & comparisons.stratum.eq(group)].set_index("baseline")
                level = 0
                for baseline in reversed(methods[:2]):
                    test = tests.loc[baseline]
                    if test.p >= 0.05:
                        continue
                    x = group_index + (methods.index(baseline) - 1) * 0.19
                    draw_comparison(ax, x, group_index + 0.19, test.p, top + 0.005 + level * 0.016, 0.004)
                    level += 1
    center = (0.082 + 0.987) / 2
    for row, title in enumerate(["Held-out UK Biobank", r"UK Biobank $\rightarrow$ CARTaGENE transfer"]):
        figure.text(center, axes[row, 0].get_position().y1 + 0.022, title, ha="center", va="bottom", fontsize=7.2)
    figure.text(0.018, 0.505, r"Incremental $R^2$", rotation=90, ha="center", va="center", fontsize=7.2)
    figure.text(center, 0.092, "Evaluation group", fontsize=6.7, ha="center", va="center")
    figure.legend(handles=[Patch(facecolor=color, label=method) for color, method in zip(compact_colors[:3], methods)],
                  loc="lower center", bbox_to_anchor=(center, 0.018), ncol=3, frameon=False,
                  fontsize=6.4, handlelength=1.8, handletextpad=0.5, columnspacing=2)
    return figure


def main():
    args = parse_args(__doc__, supplementary=True, collection="ldl_experts")
    data, comparisons = load_results(args.results)
    with plt.rc_context(style):
        print(save_figure(build_figure(data, comparisons), args.output_dir / "suppl_ldl_expert_methods.pdf"))


if __name__ == "__main__":
    main()
