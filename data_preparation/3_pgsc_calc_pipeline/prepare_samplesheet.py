# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Write a pgsc_calc samplesheet from an explicit chromosome filename template."""

import argparse
import csv
from pathlib import Path


def prepare_samplesheet(sampleset, prefix_template, genotype_format, chromosomes, output):

    if "{chrom}" not in prefix_template:
        raise ValueError("The prefix template must contain {chrom}, for example /data/chr_{chrom}.")

    if len(chromosomes) != len(set(chromosomes)):
        raise ValueError("Each chromosome must appear only once.")

    rows = []
    for chromosome in chromosomes:
        prefix = Path(prefix_template.format(chrom=chromosome)).expanduser().resolve()
        rows.append({"sampleset": sampleset, "path_prefix": str(prefix),
                     "chrom": chromosome, "format": genotype_format})

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sampleset", "path_prefix", "chrom", "format"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampleset", default="ukbb", help="Cohort label (default: ukbb).")
    parser.add_argument("--prefix-template",
                        help="Genotype prefix without extensions (default: data/{sampleset}/genotypes/chr{chrom}).")
    parser.add_argument("--format", dest="genotype_format", choices=["pfile", "bfile", "vcf"], default="pfile")
    parser.add_argument("--chromosomes", nargs="+", type=int, choices=range(1, 23), default=list(range(1, 23)))
    parser.add_argument("--output", type=Path, help="Samplesheet CSV (default: data/{sampleset}/samplesheet.csv).")
    args = parser.parse_args()
    cohort_dir = Path("data") / args.sampleset
    if args.prefix_template is None:
        args.prefix_template = str(cohort_dir / "genotypes/chr{chrom}")
    if args.output is None:
        args.output = cohort_dir / "samplesheet.csv"

    try:
        count = prepare_samplesheet(args.sampleset, args.prefix_template, args.genotype_format,
                                    args.chromosomes, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {count} chromosome rows to {args.output}")


if __name__ == "__main__":
    main()
