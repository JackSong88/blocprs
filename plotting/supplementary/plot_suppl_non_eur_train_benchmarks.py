"""Plot non-EUR-trained benchmark estimates in held-out UKB and CARTaGENE."""

import matplotlib.pyplot as plt

from ..plot_utils import draw_compact_benchmarks, save_figure, style
from ..utils import attach_compact_bootstrap, load_compact_benchmarks, parse_args


def main():
    args = parse_args(__doc__, supplementary=True)
    data, candidates = load_compact_benchmarks(args.results, non_eur=True)
    data, comparisons = attach_compact_bootstrap(data, args.results, non_eur=True)
    with plt.rc_context(style):
        figure = draw_compact_benchmarks(data, comparisons, candidates, non_eur=True)
        print(save_figure(figure, args.output_dir / "suppl_non_eur_train_benchmarks.pdf"))


if __name__ == "__main__":
    main()
