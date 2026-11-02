"""Write PLINK weight scorefile.gz files using the variants retained in both biobanks.

Windowed and full-chromosome files use the same variants, weights and matched
alleles. Only their grouping differs; full-chromosome scores are later added
to form the genome-wide PRSs.
"""

import argparse
import csv
import gzip
import math
import re
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import is_nonzero, normalize_accession


# Look up each retained variant's window and original weight for this biobank and PRS.
def load_mapping(path, cohort, full_chromosome=False):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    id_column, allele_column = f"{cohort}_id", f"{cohort}_effect_allele"

    if frame.empty or frame.duplicated(["accession", id_column, allele_column]).any():
        raise ValueError("Crosswalk must be nonempty with unique source/variant/allele entries.")

    # Reads only the matched ID, allele, PRS name, weight and assigned window into the lookup.
    frame["assigned_window"] = None if full_chromosome else frame["window_id_global"].astype("int64")
    columns = [id_column, allele_column, "accession", "effect_weight", "assigned_window"]
    mapping = defaultdict(dict)

    for variant, allele, source, weight, window in frame[columns].itertuples(index=False, name=None):
        weight = float(weight)
        mapping[(variant, allele)][normalize_accession(source)] = (window, weight)

    return mapping


# # Writes each window's scorefile with its assigned variants, effect alleles and PRS weights.
def split_scorefile(path, out_dir, mapping, full_chromosome=False):
    writers, counts = {}, defaultdict(int)
    matched = set()
    stem = path.name.removesuffix(".scorefile.gz")

    with gzip.open(path, "rt", newline="") as handle, ExitStack() as outputs:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)

        # Reads PRS names from the columns after the variant ID and effect allele.
        sources = [normalize_accession(value) for value in header[2:]]

        if len(header) < 3 or len(set(sources)) != len(sources):
            raise ValueError(f"{path}: missing or duplicated source columns.")

        for row in reader:

            if len(row) != len(header):
                raise ValueError(f"{path}: malformed score row; will not pad missing weights.")

            # Finds nonzero weights and retained PRS entries for this variant and allele.
            nonzero = [is_nonzero(value) for value in row[2:]]
            selected = mapping.get((row[0], row[1]), {})

            # Groups this variant's PRS weights by their assigned window.
            window_weights = {}

            for index, source in enumerate(sources):

                # Skips PRSs without a retained nonzero weight for this variant.
                if source not in selected or not nonzero[index]:
                    continue

                # Gets the assigned window and reference weight for this PRS entry.
                window, expected_weight = selected[source]
                contribution = (row[0], row[1], source)

                if float(row[index + 2]) != expected_weight:
                    raise ValueError(f"{path}: weight differs from the shared variant table for {contribution}.")

                if contribution in matched:
                    raise ValueError(f"{path}: repeated nonzero contribution for {contribution}.")

                # Records this variant, effect allele and PRS combination.
                matched.add(contribution)

                # Fills this PRS's weight, creating a zero-filled list if needed; +2 skips the ID and allele.
                window_weights.setdefault(window, ["0"] * len(sources))[index] = row[index + 2]

            # Creates each window or full-chromosome output and writes its header once.
            for window, values in window_weights.items():
                if window not in writers:
                    suffix = "_fullchr" if full_chromosome else f"_w{window:04d}"
                    output = out_dir / f"{stem}{suffix}.scorefile.gz"
                    stream = outputs.enter_context(gzip.open(output, "wt", compresslevel=1, newline=""))
                    writers[window] = csv.writer(stream, delimiter="\t")
                    writers[window].writerow(header)

                # Writes the variant ID, effect allele and weights in the original PRS column order.
                writers[window].writerow(row[:2] + values)
                counts[window] += 1

    return matched, counts


# Reads all pgsc_calc additive weight files for this chromosome and checks that none were missed.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biobank", required=True, choices=("ukbb", "cartagene"))
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--scorefiles", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--full-chromosome", action="store_true")
    args = parser.parse_args()

    if len(set(args.scorefiles)) != len(args.scorefiles):
        parser.error("Supply each additive scorefile once.")

    for path in args.scorefiles:
        if not re.fullmatch(rf"{args.biobank}_[0-9]+_additive_[0-9]+\.scorefile\.gz", path.name):
            parser.error(f"Unexpected matched scorefile name: {path.name}")

    mapping = load_mapping(args.crosswalk, args.biobank, args.full_chromosome)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    matched, count = set(), 0

    for path in args.scorefiles:
        links, counts = split_scorefile(path, args.out_dir, mapping, args.full_chromosome)

        print(f"{path.name}: wrote {len(counts):,} scorefiles.", flush=True)

        if matched & links:
            raise ValueError("A retained variant weight appears in more than one supplied scorefile.")

        matched.update(links)
        count += len(counts)

    # Every retained nonzero variant weight must appear once in the output.
    expected = {(variant, allele, source) for (variant, allele), sources in mapping.items()
                for source, (_, weight) in sources.items() if weight != 0}

    if expected - matched:
        raise ValueError(f"Missing {len(expected - matched)} crosswalk entries in supplied scorefiles.")

    if not count:
        raise ValueError("No retained nonzero weights were written.")

    print(f"Wrote {count} scorefiles to {args.out_dir}.", flush=True)


if __name__ == "__main__":
    main()
