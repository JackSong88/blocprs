"""Restricted-variant training-GWAS signal concentration in top-ranked windows."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import parse_args
from ..plot_utils import save_figure, style, style_axis


fractions = (.01, .025, .05, .10, .25, .50)
phenotypes = {
    "LDL": ("LDL-C", "#0072B2", "o", 3.8),
    "ASTHMA": ("Asthma", "#D55E00", "^", 4.1),
    "LOG_CRTN": ("Creatinine\u2020", "#009E73", "D", 3.5),
    "HEIGHT": ("Height", "#8C78A8", "s", 2.6),
}


# Read the dedicated analysis, not figure exports or raw SNP-level GWAS files.
def load_results(data_root, analysis_dir=None):
    names = {"LDL_MA": "LDL", "ASTHMA_MA": "ASTHMA", "LOG_CRTN_SEX": "LOG_CRTN", "HEIGHT_MA": "HEIGHT"}
    data = pd.concat([pd.read_csv(analysis_dir / name / "gwas/concentration.tsv", sep="\t")
                      for name in names], ignore_index=True)

    if not data.window_size.eq("mb1").all():
        raise ValueError("The GWAS concentration figure requires 1-Mb windows.")

    data["phenotype"] = data.phenotype.map(names)
    expected = {(phenotype, fraction) for phenotype in phenotypes for fraction in fractions}

    if data.duplicated(["phenotype", "fraction"]).any() or set(zip(data.phenotype, data.fraction)) != expected:
        raise ValueError("Incomplete or duplicate GWAS concentration results")
    if not np.isfinite(data.concentration_ratio).all() or (data.concentration_ratio < 0).any():
        raise ValueError("Invalid GWAS concentration ratios")
    if not data.cap.eq(50).all():
        raise ValueError("Expected the prespecified cap of 50 for window GWAS scores")
    if not np.allclose(data.concentration_ratio, data.signal_share / data.window_share):
        raise ValueError("Concentration must equal signal share divided by window share")

    return data


# Equal x-axis spacing represents selected fractions, not genomic distance.
def build_figure(data):
    fig, ax = plt.subplots(figsize=(183 / 25.4, 95 / 25.4))
    fig.subplots_adjust(left=.095, right=.98, bottom=.17, top=.89)
    for phenotype, (label, color, marker, size) in phenotypes.items():
        frame = data.loc[data.phenotype.eq(phenotype)].sort_values("fraction")
        ax.plot(range(len(fractions)), frame.concentration_ratio, color=color,
                linewidth=.95 if phenotype == "HEIGHT" else 1.15, marker=marker,
                markersize=size, markeredgewidth=0, label=label)

    # One-fold means the selected windows contain their proportional share of capped GWAS signal.
    ax.axhline(1, color="#666666", linewidth=.55, linestyle=(0, (1, 2)), zorder=0)
    ax.set_xlim(-.2, len(fractions)-.8)
    ax.set_ylim(0, max(1.5, float(data.concentration_ratio.max()) * 1.09))
    ax.set_xticks(range(len(fractions)), ["1", "2.5", "5", "10", "25", "50"])
    ax.set_xlabel("Top locus-specific adjustment windows (% of windows)", fontsize=7, labelpad=7)
    ax.set_ylabel("GWAS signal concentration (fold)", fontsize=7, labelpad=7)
    style_axis(ax)
    ax.tick_params(axis="both", labelsize=6.8)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.545, .995),
               ncol=4, frameon=False, fontsize=7, handlelength=1.8, handletextpad=.5, columnspacing=1.4)
    return fig


# Export the four-trait concentration comparison.
def main():
    args = parse_args(__doc__, supplementary=True, interpretation=True)
    data = load_results(args.data_root, args.analysis_dir)
    with plt.rc_context(style):
        save_figure(build_figure(data), args.output_dir / "suppl_gwas_concentration.pdf")


if __name__ == "__main__":
    main()
