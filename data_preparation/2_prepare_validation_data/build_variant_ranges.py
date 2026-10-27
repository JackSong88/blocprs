"""Collect source-weight positions for PLINK --extract range.

Output is a headerless, 1-based inclusive CHR/start/end table, not BED0.
Harmonized positions retain autosomes only; native positions retain all
chromosome labels. This position union is not the final canonical variant
panel and does not match alleles.
"""

import argparse
import gzip
from pathlib import Path

import pandas as pd


def read_header(path):
    open_file = gzip.open if path.suffix == ".gz" else open
    metadata = {}
    with open_file(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, separator, value = line.lstrip("#").partition("=")
                if separator and key.strip() in {"genome_build", "HmPOS_build"}:
                    key, value = key.strip(), value.strip()

                    if key in metadata and metadata[key] != value:
                        raise ValueError(f"{path}: conflicting {key} metadata.")

                    metadata[key] = value
                continue
            columns = line.split("\t")

            if len(columns) != len(set(columns)):
                raise ValueError(f"{path}: duplicate column names.")

            return columns, metadata
    raise ValueError(f"{path}: no weight-table header found.")


def coordinate_columns(path, genome_build):
    columns, metadata = read_header(path)
    harmonized = "hm_pos" in columns or "hm_chr" in columns
    selected = ["hm_chr", "hm_pos"] if harmonized else ["chr_name", "chr_position"]

    # Original genome_build can differ from the build of harmonized positions.
    build_key = "HmPOS_build" if harmonized else "genome_build"
    declared_build = metadata.get(build_key)

    if declared_build is not None and declared_build != genome_build:
        raise ValueError(f"{path}: {build_key}={declared_build}, expected {genome_build}.")

    return selected, harmonized


def collect_positions(frame, harmonized, loci, path):
    frame.columns = ["chromosome", "position"]
    if harmonized:

        # Unplaced harmonized variants and non-autosomal variants are omitted.
        frame = frame.dropna()
        chromosome = pd.to_numeric(frame["chromosome"], errors="coerce")
        frame = frame.loc[chromosome.isin(range(1, 23))].copy()
        frame["chromosome"] = chromosome.loc[frame.index].astype("int64").astype(str)

    elif frame.isna().any().any():
        raise ValueError(f"{path}: missing native chromosome or position.")

    chromosome = frame["chromosome"].str.strip()
    position = pd.to_numeric(frame["position"], errors="coerce")
    valid = position.notna() & position.gt(0) & position.mod(1).eq(0)

    if not valid.all() or chromosome.eq("").any() or chromosome.str.contains(r"\s").any():
        raise ValueError(f"{path}: coordinates must have chromosome labels and positive integer positions.")

    # Native chromosome labels, including X/Y/MT, are kept without filtering.
    for chrom, positions in position.groupby(chromosome):
        loci.setdefault(chrom, set()).update(int(value) for value in positions)


def build_variant_ranges(weights, genome_build, output):
    weights = list(dict.fromkeys(Path(path) for path in weights))
    output = Path(output)

    loci = {}
    for path in weights:
        columns, harmonized = coordinate_columns(path, genome_build)
        reader = pd.read_csv(path, sep="\t", comment="#", usecols=columns,
                             dtype=str, chunksize=100000)
        with reader:
            for frame in reader:

                # Enforce chromosome/position order even if input columns differ.
                collect_positions(frame[columns], harmonized, loci, path)

    total = sum(len(positions) for positions in loci.values())

    if not total:
        raise ValueError("No eligible variant positions were found.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for chromosome in sorted(loci, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c)):
            for position in sorted(loci[chromosome]):
                handle.write(f"{chromosome}\t{position}\t{position}\n")
    print(f"Wrote {total:,} unique {genome_build} positions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, required=True,
                        help="Source-weight TSV files, optionally gzip-compressed.")
    parser.add_argument("--genome-build", choices=["GRCh37", "GRCh38"], required=True,
                        help="Coordinate build; checked against relevant header metadata when present.")
    parser.add_argument("--output", type=Path, required=True,
                        help="1-based inclusive PLINK range file.")
    args = parser.parse_args()
    try:
        build_variant_ranges(args.weights, args.genome_build, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
