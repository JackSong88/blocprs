"""Plot AUROC with DeLong standard errors and paired tests by ancestry."""

import matplotlib.pyplot as plt

from ..plot_utils import draw_compact_benchmarks, save_figure, style
from ..utils import (attach_uncertainty, load_compact_benchmarks, paired_tests,
                     parse_args, phenotypes)


# Source selection is based on AUROC; uncertainty remains paired DeLong.
def load_results(path):
    data, _ = load_compact_benchmarks(path, metric="auroc")

    if not data.uncertainty_method.eq("delong").all():
        raise ValueError("The AUROC figure requires DeLong uncertainty.")

    return attach_uncertainty(data, "se"), paired_tests(data, path)


def main():
    args = parse_args(__doc__, supplementary=True, collection="discrimination")
    data, comparisons = load_results(args.results)
    with plt.rc_context(style):
        figure = draw_compact_benchmarks(data, comparisons, {"binary": phenotypes["binary"]}, auroc=True)
        print(save_figure(figure, args.output_dir / "suppl_auroc.pdf"))


if __name__ == "__main__":
    main()
