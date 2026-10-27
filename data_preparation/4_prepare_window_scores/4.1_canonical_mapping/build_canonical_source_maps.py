"""Map build-specific file rows to the same original source-variant identity."""

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import (SOURCE_MAP_COLUMNS, accession_from_source_path,
                                      first_score_header, iter_score_rows, make_source_map_row)


# Keep original alleles and weights in the key, not harmonized coordinates.
def write_source_map(source, output, build, accession):
    keys = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_MAP_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row_nr, row, _ in iter_score_rows(source):
            mapped = make_source_map_row(build, accession, row_nr, row, source)
            writer.writerow(mapped)
            keys.append(mapped["source_key"])
    duplicates = [key for key, count in Counter(keys).items() if count > 1]

    if not keys or duplicates:
        raise ValueError(f"{source}: empty source or duplicate source keys: {duplicates[:3]}")

    return keys


# Require both coordinate builds, while allowing build-specific retained variants.
def build_source_maps(source_root, out_root):
    sources = {}
    for build in ("GRCh37", "GRCh38"):
        sources[build] = {}
        for path in sorted((source_root / build).glob("*.txt.gz")):
            accession = accession_from_source_path(path)

            if accession in sources[build]:
                raise ValueError(f"Duplicate source filename identity: {accession}")

            sources[build][accession] = path

    if not sources["GRCh37"] or sources["GRCh37"].keys() != sources["GRCh38"].keys():
        raise ValueError("Supply the same source accessions under GRCh37/ and GRCh38/.")

    for accession in sorted(sources["GRCh37"]):
        keys = {}
        for build in ("GRCh37", "GRCh38"):
            output = out_root / build / f"{accession}.source_map.tsv.gz"
            keys[build] = write_source_map(sources[build][accession], output, build, accession)

        # Without rsIDs, source coordinates and row order identify the same variants.
        if "rsID" not in first_score_header(sources["GRCh37"][accession]):

            if keys["GRCh37"] != keys["GRCh38"]:
                raise ValueError(f"{accession}: original source coordinates/order differ between builds.")

        shared = len(set(keys["GRCh37"]) & set(keys["GRCh38"]))

        if not shared:
            raise ValueError(f"{accession}: no shared original source identities.")

        print(f"{accession}: {shared:,} shared source identities.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    build_source_maps(args.source_root, args.out_root)


if __name__ == "__main__":
    main()
