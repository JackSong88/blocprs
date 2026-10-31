"""Plot UKBB model progression from raw evaluations and bootstrap analysis."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .plot_utils import method_colors, save_figure, style, style_axis
from .utils import display_phenotype, load_progression, parse_args


CONTINUOUS = [
    "LDL_MA", "DBP_MA", "HEIGHT_MA", "LOG_BMI_MA", "URT_SEX", "WHR_SEX",
    "LOG_CRTN_SEX", "LOG_HDL_MA", "TC_MA", "FEV1_FVC_MA", "SBP_MA",
    "LOG_TG_MA", "LOG_TST_SEX",
]
BINARY = [
    "ASTHMA_MA", "CAD_MT", "AF_MT", "ASTHMA_MT", "GOUT_MT", "HF_MT",
    "HTN_MT", "STR_MT", "T1D_MT", "T2D_MA", "T2D_MT", "DEM_MT",
]
EXCLUDED = {"SBP_ADJ_MA", "DBP_ADJ_MA", "TC_ADJ_MA", "LDL_ADJ_MA"}
METHODS = [
    "Best single PRS (oracle)", "Best single-source BLocPRS (oracle)",
    "MultiPRS", "MultiPRS + regional scaling", "BLocPRS (mb1)",
]
LABELS = [
    "Best single-source PRS", "Best single-source BLocPRS", "MultiPRS",
    "BLocPRS (fixed composition)", "BLocPRS",
]
MARKERS = ["o", "o", "s", "^", "D"]
MARKER_SIZES = [3.0, 3.0, 2.8, 3.1, 2.5]


# These labels preserve the source figure's trait and study distinctions.
DISPLAY_LABELS = {
    "LOG_TST_SEX": "TST\u2020 (SEX)", "FEV1_FVC_MA": "FEV1/FVC",
    "TC_MA": "TC", "WHR_SEX": "WHR (SEX)", "LDL_MA": "LDL-C",
    "LOG_CRTN_SEX": "CRTN\u2020 (SEX)", "LOG_TG_MA": "TG\u2020",
    "LOG_HDL_MA": "HDL-C\u2020", "URT_SEX": "URT (SEX)", "SBP_MA": "SBP",
    "DBP_MA": "DBP", "LOG_BMI_MA": "BMI\u2020", "HEIGHT_MA": "Height",
    "DEM_MT": "DEM", "GOUT_MT": "GOUT", "CAD_MT": "CAD", "STR_MT": "STR",
    "T1D_MT": "T1D", "ASTHMA_MA": "Asthma (MA)", "ASTHMA_MT": "Asthma (RT)",
    "T2D_MA": "T2D (MA)", "HF_MT": "HF", "AF_MT": "AF", "HTN_MT": "HTN",
    "T2D_MT": "T2D (RT)",
}


# Both source maxima are selected independently in the held-out UKB cohort.
def load_values(path):
    data = load_progression(path, "ukbb")
    labels = dict(zip(["Best single-source PRS", "Best regional single-source PRS",
                       "Nonwindowed MultiPRS", "Fixed-composition regional MultiPRS",
                       "BLocPRS mb1"], METHODS))
    data["method"] = data.method.map(labels)
    return data.rename(columns={"phenotype": "Phenotype", "family": "Family", "method": "Model",
                                "value": "Value", "n": "N", "source": "Source"})


def plot_figure(data):
    figure_style = {**style, "axes.labelsize": 6.9, "xtick.labelsize": 6.4,
                    "ytick.labelsize": 6.7, "legend.fontsize": 7.0}
    with mpl.rc_context(figure_style):
        fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 135 / 25.4))
        fig.subplots_adjust(left=0.112, right=0.985, top=0.905, bottom=0.082, wspace=0.28)
        families = [("continuous", r"Incremental $R^2$"), ("binary", r"Nagelkerke $R^2$")]
        for column, (family, metric) in enumerate(families):
            ax = axes[column]
            panel = data.loc[data["Family"].eq(family)]

            # Use one BLocPRS-based trait order for all five methods within each outcome family.
            phenotypes = (panel.loc[panel["Model"].eq(METHODS[-1])]
                          .sort_values(["Value", "Phenotype"], kind="mergesort")["Phenotype"].tolist())
            spacing = 1.05
            y = np.arange(len(phenotypes), dtype=float) * spacing
            for index in range(0, len(phenotypes), 2):
                ax.axhspan(y[index] - spacing / 2, y[index] + spacing / 2,
                           facecolor="#F6F7F7", edgecolor="none", zorder=0)
            indexed = panel.set_index(["Model", "Phenotype"])
            for index, method in enumerate(METHODS):
                values = indexed.loc[method].reindex(phenotypes)
                color = method_colors[LABELS[index]]

                # Fixed offsets separate methods; intervals are +/- one bootstrap SE.
                ax.errorbar(values["Value"], y + (index - 2) * 0.19,
                            xerr=values["bootstrap_se"], fmt=MARKERS[index], linestyle="none",
                            markersize=MARKER_SIZES[index], markeredgecolor=color,
                            markerfacecolor="white" if index == 0 else color,
                            markeredgewidth=0.55, ecolor=color, elinewidth=0.55,
                            capsize=0, alpha=1.0, zorder=3)
            minimum = min(0.0, float((panel["Value"] - panel["bootstrap_se"]).min()))
            maximum = float((panel["Value"] + panel["bootstrap_se"]).max())
            ax.set_xlim(minimum * 1.055, maximum * 1.055)
            ax.xaxis.set_major_locator(MultipleLocator(0.05))
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda value, _: "0" if np.isclose(value, 0, atol=5e-5) else f"{value:.2f}"))
            ax.set_ylim(-0.55 * spacing, y[-1] + 0.55 * spacing)
            ax.invert_yaxis()
            ax.set_yticks(y, [DISPLAY_LABELS.get(p, display_phenotype(p)) for p in phenotypes])
            ax.set_xlabel(metric, labelpad=3.2)
            style_axis(ax)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=2.5, width=0.55, pad=2.2)
            ax.tick_params(axis="x", length=3.0, width=0.65, pad=1.7)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#52595C")
                ax.spines[spine].set_linewidth(0.55)
            ax.text(0, 1.025, "AB"[column], transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8.4, fontweight="bold", color="#222222", clip_on=False)
        handles = [Line2D([], [], linestyle="none", marker=MARKERS[index],
                          markersize=MARKER_SIZES[index] + 0.8,
                          markerfacecolor="white" if index == 0 else method_colors[label],
                          markeredgecolor=method_colors[label], markeredgewidth=0.55, label=label)
                   for index, label in enumerate(LABELS)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncols=5,
                   frameon=False, handlelength=0.85, handleheight=0.90, handletextpad=0.42,
                   columnspacing=0.85, borderaxespad=0, fontsize=7.0)
    return fig


def main():
    args = parse_args(__doc__, collection="progression")
    data = load_values(args.results)
    save_figure(plot_figure(data), args.output_dir / "Figure_4.pdf")


if __name__ == "__main__":
    main()
