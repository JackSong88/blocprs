"""Retain canonical variants and partition matched weights into windows or full chromosomes."""

import argparse
import csv
import gzip
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import is_nonzero, normalize_accession


# Map each cohort-local variant/allele/source combination to its reference window.
def load_mapping(path, cohort, full_chromosome=False):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    id_column, allele_column = f"{cohort}_id", f"{cohort}_effect_allele"
    keys = ["accession", id_column, allele_column]

    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Crosswalk must be nonempty with unique source/variant/allele entries.")

    mapping = defaultdict(dict)
    for row in frame.to_dict("records"):
        window = None if full_chromosome else int(row["window_id_global"])
        weight = float(row["effect_weight"])

        if not math.isfinite(weight):
            raise ValueError("Crosswalk weights must be finite.")

        mapping[(row[id_column], row[allele_column])][normalize_accession(row["accession"])] = (window, weight)
    return mapping


# Preserve matched alleles and exact weight strings; zeros exclude other source/window contributions.
def split_scorefile(path, out_dir, mapping, full_chromosome=False):
    handles, writers, counts = {}, {}, defaultdict(int)
    matched = set()
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sources = [normalize_accession(value) for value in header[2:]]

        if len(header) < 3 or len(set(sources)) != len(sources):
            raise ValueError(f"{path}: missing or duplicated source columns.")

        try:
            for row in reader:

                if len(row) != len(header):
                    raise ValueError(f"{path}: malformed score row; will not pad missing weights.")

                nonzero = [is_nonzero(value) for value in row[2:]]
                selected = mapping.get((row[0], row[1]), {})
                values_by_window = {}
                for index, source in enumerate(sources):
                    if source not in selected:
                        continue
                    window, expected_weight = selected[source]
                    link = (row[0], row[1], source)
                    if nonzero[index]:

                        if float(row[index + 2]) != expected_weight:
                            raise ValueError(f"{path}: weight differs from the canonical crosswalk for {link}.")

                        if link in matched:
                            raise ValueError(f"{path}: repeated nonzero contribution for {link}.")

                        matched.add(link)
                        values_by_window.setdefault(window, ["0"] * len(sources))[index] = row[index + 2]
                for window, values in values_by_window.items():
                    if window not in writers:
                        stem = path.name.removesuffix(".scorefile.gz")
                        suffix = "_fullchr" if full_chromosome else f"_w{window:04d}"
                        output = out_dir / f"{stem}{suffix}.scorefile.gz"
                        stream = gzip.open(output, "wt", compresslevel=1, newline="")
                        handles[window] = stream
                        writers[window] = csv.writer(stream, delimiter="\t")
                        writers[window].writerow(header)
                    writers[window].writerow(row[:2] + values)
                    counts[window] += 1
        finally:
            for stream in handles.values():
                stream.close()
    return matched, counts


# Additive partitions are supplied explicitly; an absent file cannot be mistaken for a zero contribution.
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

        if matched & links:
            raise ValueError("A nonzero canonical contribution appears in multiple additive partitions.")

        matched.update(links)
        count += len(counts)
    expected = {(variant, allele, source) for (variant, allele), sources in mapping.items()
                for source, (_, weight) in sources.items() if weight != 0}

    if expected - matched:
        raise ValueError(f"Missing {len(expected - matched)} crosswalk entries in supplied scorefiles.")

    if not count:
        raise ValueError("No nonzero canonical weights were written.")

    print(f"Wrote {count} scorefiles to {args.out_dir}.")


if __name__ == "__main__":
    main()
