"""Saves a lookup table for each original PRS weight file.

Each row contains the variant's row number, key, coordinates, alleles and weight.
The next script uses the row numbers in pgsc_calc's matching logs to look up
these original variant details.
"""

import argparse
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import (SOURCE_MAP_COLUMNS, accession_from_source_path,
                                      first_score_header, iter_score_rows, make_source_map_row)


# Saves each variant's row number, key, coordinates, alleles and weight.
def write_source_map(source, output, build, accession):
    keys = []
    output.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_MAP_COLUMNS, delimiter="\t")
        writer.writeheader()

        # Key: PRS|key_type|variant_id|effect_allele|other_allele|effect_weight
        # Illustrative examples:
        #   PGS000001|rsid|rs123|A|G|0.02
        #   PGS000001|source_chrpos|1:12345|A|G|0.02  (without an rsID)
        for row_nr, row, _ in iter_score_rows(source):
            mapped = make_source_map_row(build, accession, row_nr, row)
            writer.writerow(mapped)
            keys.append(mapped["source_key"])

    # Stops if the weight file is empty or more than one variant entry has the same key.
    duplicates = [key for key, count in Counter(keys).items() if count > 1]

    if not keys or duplicates:
        raise ValueError(f"{source}: empty source or duplicate source keys: {duplicates[:3]}")

    return keys


# Finds each PRS's GRCh37 and GRCh38 weight files and writes a lookup table for each.
def build_source_maps(source_root, out_root):
    sources = {}

    # Collects the weight files by PRS name in each build folder.
    for build in ("GRCh37", "GRCh38"):
        sources[build] = {}

        for path in sorted((source_root / build).glob("*.txt.gz")):
            accession = accession_from_source_path(path)

            if accession in sources[build]:
                raise ValueError(f"Multiple weight files have the PRS name {accession}.")

            sources[build][accession] = path

    # Checks that both folders contain the same PRSs; their variant sets may differ.
    if not sources["GRCh37"] or sources["GRCh37"].keys() != sources["GRCh38"].keys():
        raise ValueError("Supply the same PRSs under GRCh37/ and GRCh38/.")

    # Numbers variants separately in each file; the two files may list variants in different orders.
    for accession in sorted(sources["GRCh37"]):
        keys = {}

        for build in ("GRCh37", "GRCh38"):
            print(f"Mapping original weight entries: {accession} {build}.", flush=True)
            output = out_root / build / f"{accession}.source_map.tsv.gz"
            keys[build] = write_source_map(sources[build][accession], output, build, accession)

        # Requires the same keys in the same order if the GRCh37 file has no rsID column.
        if "rsID" not in first_score_header(sources["GRCh37"][accession]) and keys["GRCh37"] != keys["GRCh38"]:
            raise ValueError(f"{accession}: original coordinates/order differ between builds.")

        # Counts keys present in both weight files, before checking either biobank's genotypes.
        shared = len(set(keys["GRCh37"]) & set(keys["GRCh38"]))

        if not shared:
            raise ValueError(f"{accession}: no shared variant entries in the original PRS weight files.")

        print(f"{accession}: {shared:,} shared variant entries.", flush=True)


# Reads the input and output paths, then creates the lookup tables.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    build_source_maps(args.source_root, args.out_root)


if __name__ == "__main__":
    main()
