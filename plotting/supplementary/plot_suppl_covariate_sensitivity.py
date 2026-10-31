"""Stored-versus-expanded covariate sensitivity from matched 1-Mb ablation evaluations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import core_phenotypes, metric_labels, metric_tokens, one_row, parse_args, phenotype_label, read_results, select_results
from ..plot_utils import save_pages, style, style_axis


designs = ("standard", "expanded")
cohorts = {"ukbb_native": "Held-out UK Biobank", "full_cag_transfer": "UKB to CaG transfer"}
analyses = core_phenotypes + [(name, "continuous") for name in (
    "LDL_ADJ_MA", "TC_ADJ_MA", "DBP_ADJ_MA", "SBP_ADJ_MA")]


# Require both covariate designs for each frozen-model component.
def load_results(path):
    data = read_results(path)
    records = []
    for scenario, cohort in zip(cohorts, ("ukbb", "cartagene")):
        for phenotype, family in analyses:

            if cohort == "cartagene" and phenotype in {"LOG_TST_SEX", "DEM_MT"}:
                continue

            frame = select_results(data, cohort=cohort, phenotype=phenotype, group="all",
                                   train_group="all", train_size="full", panel="main",
                                   window_mb=1, metric=metric_tokens[family])
            counts = []
            for design in designs:
                for method, mode in (("anchor_only", "anchor_only"), ("blocprs", "full")):
                    row = one_row(frame, covariate_set=design, method=method)
                    counts.append(row.n)
                    records.append(dict(Scenario=scenario, Phenotype=phenotype, family=family,
                                        NullDesign=design, Mode=mode, Value=row.value, n=row.n))

            if len(set(counts)) != 1:
                raise ValueError("Covariate sensitivity must use the same participants across designs.")

    return pd.DataFrame(records)


# Pair the frozen anchor and complete genetic score under each covariate design.
def build_pages(data):
    for scenario, cohort in cohorts.items():
        for family in ("continuous", "binary"):
            frame = data.loc[data.Scenario.eq(scenario) & data.family.eq(family)]
            pivot = frame.pivot(index="Phenotype", columns=["NullDesign", "Mode"], values="Value")
            fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, max(80, len(pivot)*9+30) / 25.4))
            fig.subplots_adjust(left=.19, right=.96, bottom=.18, top=.83, wspace=.28)
            for ax, design, title in zip(axes, designs, ("Stored covariates", "Expanded covariates")):
                anchor, full = pivot[(design, "anchor_only")], pivot[(design, "full")]
                y = np.arange(len(pivot))
                ax.hlines(y, anchor, full, color="#B0B0B0", linewidth=.8)
                ax.scatter(anchor, y, color="#6BAED6", s=12, label="Genome-wide component")
                ax.scatter(full, y, color="#6A51A3", s=12, label="Effective")
                ax.set_yticks(y, [phenotype_label(phenotype) for phenotype in pivot.index], fontsize=6)
                ax.invert_yaxis()
                style_axis(ax)
                ax.set_xlabel(metric_labels[family], fontsize=7)
                ax.set_title(title, fontsize=7)
            axes[1].tick_params(labelleft=False)
            fig.suptitle(f"{cohort} - {family}", fontsize=8, y=.96)
            fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=6)
            yield fig


# Missing expanded-design analyses stop export; no evaluation is performed here.
def main():
    args = parse_args(__doc__, supplementary=True, collection="covariate_sensitivity")
    data = load_results(args.results)
    with plt.rc_context(style):
        save_pages(build_pages(data), args.output_dir / "suppl_covariate_sensitivity.pdf")


if __name__ == "__main__":
    main()
