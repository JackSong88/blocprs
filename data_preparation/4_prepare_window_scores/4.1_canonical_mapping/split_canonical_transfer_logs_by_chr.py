#!/usr/bin/env python3
"""Adds original PRS variant details to pgsc_calc's accepted matches.

Reads each biobank's matching log in chunks. Uses row numbers to look up the
original variant keys, alleles and weights, then saves one table per chromosome.
The next script joins the UKB and CaG tables using those keys.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from time import monotonic

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import (ANNOTATED_LOG_COLUMNS, normalize_accession,
                                      normalize_chr, parse_chromosomes)


LOG_COLUMNS = [
    "row_nr",
    "accession",
    "chr_name",
    "chr_position",
    "effect_allele",
    "other_allele",
    "effect_weight",
    "effect_type",
    "ID",
    "matched_effect_allele",
    "best_match",
    "exclude",
    "match_status",
]


SOURCE_COLUMNS = [
    "row_nr",
    "source_key",
    "source_key_type",
    "source_variant_id",
    "source_chr",
    "source_position",
    "source_effect_allele",
    "source_other_allele",
    "source_effect_weight",
]


# Treats 1, true, t and yes as true, regardless of capitalization.
def truthy(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"1", "true", "t", "yes"})


# Keeps pgsc_calc's accepted matches and adds the original variant keys, alleles and weights.
def annotate_chunk(chunk: pd.DataFrame, load_source_map,
                   chromosomes: set[str]) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["chr_name"] = chunk["chr_name"].map(normalize_chr)
    chunk["chr_position"] = pd.to_numeric(chunk["chr_position"], errors="coerce")
    chunk["row_nr"] = pd.to_numeric(chunk["row_nr"], errors="coerce")

    # Keeps pgsc_calc's non-excluded best matches on the requested chromosomes.
    # Uses pgsc_calc's allele matches without repeating allele matching here.
    keep = (chunk["chr_name"].isin(chromosomes)
            & chunk["chr_position"].notna() & chunk["row_nr"].notna()
            & chunk["match_status"].fillna("").astype(str).str.lower().eq("matched")
            & truthy(chunk["best_match"]) & ~truthy(chunk["exclude"]))
    chunk = chunk.loc[keep].copy()

    if chunk.empty:
        return pd.DataFrame(columns=ANNOTATED_LOG_COLUMNS)

    chunk["accession"] = chunk["accession"].map(normalize_accession)
    chunk["row_nr"] = chunk["row_nr"].astype("int64")
    chunk["chr_position"] = chunk["chr_position"].astype("int64").astype(str)
    chunk = chunk.rename(columns={"ID": "id"}).fillna("")

    annotated = []

    for accession, group in chunk.groupby("accession", sort=False):
        source_map = load_source_map(accession)

        # Joins pgsc_calc matches to this build's original PRS entries using row_nr.
        merged = group.merge(source_map, on="row_nr", how="left", validate="many_to_one")

        if merged["source_key"].isna().any():
            raise ValueError(f"{accession}: accepted matches are missing from the PRS weight lookup table.")

        # Checks that allele labels and weight text match the original PRS file exactly.
        mismatch = (merged["effect_allele"].astype(str).ne(merged["source_effect_allele"].astype(str))
                    | merged["other_allele"].astype(str).ne(merged["source_other_allele"].astype(str))
                    | merged["effect_weight"].astype(str).ne(merged["source_effect_weight"].astype(str)))

        if mismatch.any():
            raise ValueError(f"{accession}: matching-log alleles or weights differ from the original PRS file.")

        annotated.append(merged)

    out = pd.concat(annotated, ignore_index=True)

    return out[ANNOTATED_LOG_COLUMNS]


# Reads one biobank's log in chunks, writes separate chromosome files and counts their rows.
def split_one_log(log_path: Path, biobank: str, build: str, source_map_root: Path,
                  out_dir: Path, chromosomes: list[str], chunksize: int,
                  verbose: bool = False) -> dict[str, int]:

    # Keeps the three most recently used PRS tables in memory for this biobank.
    @lru_cache(maxsize=3)
    def load_source_map(accession):
        path = source_map_root / build / f"{accession}.source_map.tsv.gz"
        frame = pd.read_csv(path, sep="\t", usecols=SOURCE_COLUMNS, dtype=str, keep_default_na=False)
        frame["row_nr"] = pd.to_numeric(frame["row_nr"], errors="raise").astype("int64")

        return frame

    chromosome_set = set(chromosomes)
    counts = dict.fromkeys(chromosomes, 0)
    total_read = 0
    last_report = monotonic()
    print(f"{biobank}: reading and annotating matched variants from {log_path}.", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keeps chromosome output files open across chunks, then closes them automatically.
    with ExitStack() as files:
        outputs = {}

        for chrom in chromosomes:
            path = out_dir / f"{biobank}_chr{chrom}.tsv.gz"
            outputs[chrom] = files.enter_context(gzip.open(path, "wt", newline=""))
            writer = csv.writer(outputs[chrom], delimiter="\t", lineterminator="\n")
            writer.writerow(ANNOTATED_LOG_COLUMNS)

        # Reads one log chunk at a time and reuses the cached PRS tables.
        reader = pd.read_csv(log_path, compression="gzip", usecols=LOG_COLUMNS,
                             dtype=str, chunksize=chunksize, low_memory=False)

        for chunk_idx, chunk in enumerate(reader, start=1):
            total_read += len(chunk)
            annotated = annotate_chunk(chunk, load_source_map, chromosome_set)

            for chrom, group in annotated.groupby("chr_name", sort=False):
                group.to_csv(outputs[chrom], sep="\t", index=False, header=False)
                counts[chrom] += len(group)

            now = monotonic()
            if verbose or now - last_report >= 30:
                print(f"{biobank}: chunk {chunk_idx:,}, read {total_read:,}, "
                      f"retained {sum(counts.values()):,} matched variants.", flush=True)
                last_report = now

    return counts


# Uses GRCh37 original weights for the UKB log and GRCh38 original weights for the CaG log.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ukbb-log", type=Path, required=True)
    parser.add_argument("--cartagene-log", type=Path, required=True)
    parser.add_argument("--source-map-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chromosomes", default="1-22")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    chromosomes = parse_chromosomes(args.chromosomes)

    assert chromosomes
    assert args.chunksize >= 1

    # Writes each biobank's files from the beginning on every run.
    for cohort, build, path in (("ukbb", "GRCh37", args.ukbb_log),
                                ("cartagene", "GRCh38", args.cartagene_log)):
        counts = split_one_log(path, cohort, build, args.source_map_root,
                               args.out_dir, chromosomes, args.chunksize, args.verbose)

        print(f"{cohort}: retained {sum(counts.values()):,} matched variants.", flush=True)


if __name__ == "__main__":
    main()
