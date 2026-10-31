"""Plot normalized non-EUR benchmarks for full and non-EUR UKB training."""

import matplotlib.pyplot as plt
import numpy as np

from ..plot_utils import draw_matrix, save_figure, score_color, score_size, style
from ..utils import load_benchmark, parse_args


panels = [("ukbb", False, r"Full UKB training $\rightarrow$ non-EUR UKB held-out"),
          ("cartagene", False, r"Full UKB training $\rightarrow$ non-EUR CaG transfer"),
          ("ukbb", True, r"Non-EUR UKB training $\rightarrow$ non-EUR UKB held-out"),
          ("cartagene", True, r"Non-EUR UKB training $\rightarrow$ non-EUR CaG transfer")]


# Load the same prespecified performance and case-count eligibility as Figure 2.
def load_inputs(data_root):
    return [load_benchmark(data_root, cohort, non_eur_eval=True, non_eur_train=training)
            for cohort, training, _ in panels]


def build_figure(results):
    figure = plt.figure(figsize=(183 / 25.4, 200 / 25.4))
    grid = figure.add_gridspec(4, 4, left=0.028, right=0.985, bottom=0.11, top=0.92,
                              width_ratios=[1.50, 0.58, 0.07, 7.6], hspace=0.42, wspace=0)
    for row, (data, (_, _, title)) in enumerate(zip(results, panels)):
        row_grid = [grid[row, column] for column in range(4)]
        draw_matrix(figure, row_grid, data, "ABCD"[row], title)
    ax = figure.add_axes([0.028, 0.018, 0.965, 0.064])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.25, 0.85, "Summary score (% of best-performing method)", ha="center", fontsize=6)
    colors = np.array([score_color("overall", value) for value in np.linspace(0, 1, 256)])
    ax.imshow(colors.reshape(1, 256, 3), extent=(0.16, 0.34, 0.38, 0.56), aspect="auto")
    for x, label in [(0.16, "0%"), (0.25, "50%"), (0.34, "100%")]:
        ax.text(x, 0.15, label, ha="center", fontsize=5.6)
    ax.text(0.75, 0.85, "Phenotype score (% of best-performing method)", ha="center", fontsize=6)
    for x, value in [(0.67, 0.05), (0.75, 0.50), (0.83, 1.0)]:
        ax.scatter(x, 0.47, s=score_size(value), color=score_color("continuous", value),
                   edgecolor="#3F3F3F", linewidth=0.35)
        ax.text(x, 0.15, f"{value:.0%}", ha="center", fontsize=5.6)
    return figure


def main():
    args = parse_args(__doc__, supplementary=True)
    results = load_inputs(args.results)
    with plt.rc_context(style):
        print(save_figure(build_figure(results), args.output_dir / "suppl_non_eur_circle_benchmarks.pdf"))


if __name__ == "__main__":
    main()
