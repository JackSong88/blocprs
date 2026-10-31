"""Combined-window SD ratios under magnitude and signed-covariance rankings."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import parse_args
from ..plot_utils import reference_bins, save_figure, style, style_axis


def load_results(data_root, analysis_dir=None):
    from analyses.interpretation.contributions import covariance_bins

    windows = pd.concat([pd.read_csv(path, sep="\t") for path in sorted(analysis_dir.glob("*/windows.tsv"))],
                        ignore_index=True)
    windows = windows.loc[~windows.phenotype.str.contains("_ADJ_")]

    if not windows.window_size.eq("mb1").all():
        raise ValueError("The covariance sensitivity figure requires 1-Mb windows.")

    values = covariance_bins(windows)
    values = values.loc[values.ranking.isin(("original", "signed_covariance"))]

    # Missing bins cannot silently change the set of phenotype means.
    if (values.phenotype.nunique() != 25 or len(values) != 350
            or values.duplicated(["phenotype", "ranking", "bin"]).any()
            or not np.isfinite(values.net_sd_ratio.dropna()).all() or values.net_sd_ratio.dropna().lt(0).any()):
        raise ValueError("Incomplete or invalid covariance-sensitivity summaries")

    for _, group in values.groupby(["phenotype", "ranking"]):
        if set(group.bin) != set(reference_bins):
            raise ValueError("Incomplete reference-ranked bins")

    # Average phenotype-level ratios, rather than pooling windows across phenotypes.
    return values.groupby(["ranking", "bin"]).net_sd_ratio.mean().unstack("bin")[reference_bins]


def build_figure(values):
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 86/25.4), sharey=True, layout="constrained")
    for ax, rank, title, letter, color in zip(axes, ["original", "signed_covariance"],
            ["Reference magnitude ranking", "Reference covariance ranking"], "ab", ["#666666", "#D55E00"]):
        ax.plot(range(7), values.loc[rank], color=color, lw=1.3, marker="o", ms=3.2)
        ax.set_xticks(range(7), reference_bins, rotation=35, ha="right")
        ax.set_title(title, fontsize=7.5, pad=9)
        ax.text(-.08, 1.065, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
        ax.set_xlabel("Reference-ranked percentile bin", fontsize=7, labelpad=5)
        ax.set_ylabel("Locus-specific adjustment\n(% of reference contribution)", fontsize=7, labelpad=6)
        ax.tick_params(labelsize=7, labelleft=True)
        style_axis(ax, grid_axis="y")
    axes[0].set_ylim(0, values.to_numpy().max()*1.12)
    return fig


def main():
    args = parse_args(__doc__, supplementary=True, interpretation=True)
    values = load_results(args.data_root, args.analysis_dir)
    with plt.rc_context(style):
        save_figure(build_figure(values), args.output_dir / "suppl_adjustment_covariance_sensitivity.pdf")


if __name__ == "__main__":
    main()
