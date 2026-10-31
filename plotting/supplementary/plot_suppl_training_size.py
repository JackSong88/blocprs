"""Plot all core-phenotype training-size trajectories in UKB and CARTaGENE."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from ..plot_utils import save_figure, style, style_axis
from ..utils import load_training_sizes, metric_labels, parse_args, phenotype_label


sizes = [2500, 5000, 12500, 25000, 50000, 100000, "full"]
size_labels = ["2.5k", "5k", "12.5k", "25k", "50k", "100k", "Full"]
models = ["Nonwindowed MultiPRS", "BLocPRS mb1"]
labels = ["MultiPRS", "BLocPRS (1 Mb)"]
colors = ["#6BAED6", "#6A51A3"]


# Unlike the normalized summaries, these trajectories have no performance eligibility filter.
def load_results(path):
    return load_training_sizes(path)


# Equal vertical spans preserve the visual size of each absolute change in R2.
def build_figure(data, family):
    frame = data.loc[data.family.eq(family)]
    ranges = frame.groupby(["phenotype", "cohort"]).value.agg(["min", "max"])

    # Round outward to 0.01 R2 steps and use the widest span in this outcome family.
    lower_steps = np.floor((ranges["min"] - 0.001) / 0.01).astype(int)
    upper_steps = np.ceil((ranges["max"] + 0.001) / 0.01).astype(int)
    span_steps = max(2, int((upper_steps - lower_steps).max()))
    order = frame.phenotype.drop_duplicates().tolist()
    nrows = (len(order) + 2) // 3
    per_row, extra = divmod(len(order), nrows)
    row_sizes = [per_row + (row < extra) for row in range(nrows)]
    positions = [(row, column + (3 - count) / 2)
                 for row, count in enumerate(row_sizes) for column in range(count)]
    height = nrows * 29 + 35
    figure = plt.figure(figsize=(183 / 25.4, height / 25.4))
    for block, (cohort, title) in enumerate([
            ("ukbb", "Held-out UK Biobank"), ("cartagene", r"UK Biobank $\rightarrow$ CARTaGENE transfer")]):
        left = 12 + block * 87
        figure.text((left + 40.5) / 183, 1 - 10 / height, title, fontsize=7, ha="center")
        figure.text((left - 5) / 183, 1 - 10 / height, "AB"[block], fontsize=8.5, fontweight="bold")
        for phenotype, (row, column) in zip(order, positions):
            ax = figure.add_axes([(left + column * 28) / 183, 1 - (17 + row * 29 + 21) / height,
                                  20 / 183, 21 / height])
            selected = frame.loc[frame.phenotype.eq(phenotype) & frame.cohort.eq(cohort)]
            for model, color, marker, linestyle in zip(models, colors, ["o", "s"], ["--", "-"]):
                values = selected.loc[selected.model.eq(model)].set_index("size").reindex(sizes).value
                ax.plot(range(7), values, color=color, marker=marker, markersize=1 if marker == "s" else 1.35,
                        linewidth=0.75, linestyle=linestyle)
            style_axis(ax)

            # Panel origins differ; unavailable CaG traits retain the observed UKB origin.
            key = (phenotype, cohort)
            origin = int(lower_steps.loc[key] if key in lower_steps.index else lower_steps.loc[phenotype].min())
            ticks = np.linspace(origin, origin + span_steps, 3) / 100
            ax.set(ylim=(ticks[0], ticks[-1]), yticks=ticks)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_xticks(range(7), size_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
            ax.tick_params(axis="x", labelbottom=row == nrows - 1)
            ax.tick_params(axis="y", labelsize=6)
            ax.set_title(phenotype_label(phenotype).replace("\n", " "), fontsize=6.8, pad=2)
            if selected.empty:
                ax.text(0.5, 0.45, "\u2014", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#777777")
    figure.supylabel(metric_labels[family], x=0.006, fontsize=7)
    figure.supxlabel("Training sample size", y=15 / height, fontsize=7)
    handles = [Line2D([], [], color=color, marker=marker, markersize=2 if marker == "s" else 2.5, linestyle=linestyle)
               for color, marker, linestyle in zip(colors, ["o", "s"], ["--", "-"])]
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 6 / height),
                  ncol=2, fontsize=6.5, frameon=False)
    return figure


def main():
    args = parse_args(__doc__, supplementary=True, collection="sample_size")
    data = load_results(args.results)
    with plt.rc_context(style):
        for family in ["continuous", "binary"]:
            print(save_figure(build_figure(data, family), args.output_dir / f"suppl_training_size_{family}.pdf"))


if __name__ == "__main__":
    main()
