"""Combine a saved model and canonical or harmonized source SNP weights into one TSV."""

import argparse
import csv
import gzip
from pathlib import Path

import pandas as pd

from blocprs import BLocPRS, export_snp_weights
from blocprs.canonical_export import export_canonical_snp_weights


# Read source labels and variant identifiers without converting them to numbers.
def read_source_weights(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    # Check headers before pandas renames duplicate columns.
    with opener(path, "rt", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"))

    if len(header) != len(set(header)):
        raise ValueError("Source weights contain duplicate column names.")

    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


# Apply frozen genetic coefficients using generic weights or the assigned canonical crosswalk.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Saved BLocPRS model archive.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-weights", type=Path,
                        help="Complete harmonized source-weight TSV or TSV.gz with window labels.")
    inputs.add_argument("--cross-mapping", type=Path, nargs="+",
                        help="Final window-assigned canonical crosswalk files, e.g. cross_mapping_chr*.tsv.gz.")
    parser.add_argument("--cohort", choices=("ukbb", "cartagene"), help="Cohort-local alleles and IDs for canonical export.")
    parser.add_argument("--genome-build", choices=("GRCh37", "GRCh38"), help="Actual genotype build; required for canonical export.")
    parser.add_argument("--pvar-template", help="Cohort PVAR path with {chr}; default: data/<cohort>/genotypes/chr{chr}.pvar.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Combined plain-text TSV, including genome build and centering offset.")
    args = parser.parse_args(argv)

    if args.cross_mapping and (args.cohort is None or args.genome_build is None):
        parser.error("--cross-mapping requires --cohort and --genome-build.")

    if args.source_weights and any(value is not None for value in (args.cohort, args.genome_build, args.pvar_template)):
        parser.error("--cohort, --genome-build and --pvar-template apply only to --cross-mapping.")

    model = BLocPRS.load(args.model)
    if args.cross_mapping:
        combined = export_canonical_snp_weights(model, args.cross_mapping, cohort=args.cohort,
                                               genome_build=args.genome_build, pvar_template=args.pvar_template)
    else:
        source_weights = read_source_weights(args.source_weights)
        combined = export_snp_weights(model, source_weights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(args.output)
    print(f"variants={len(combined.weights)} build={combined.genome_build} "
          f"offset={combined.offset:.17g} saved={args.output}")


if __name__ == "__main__":
    main()
