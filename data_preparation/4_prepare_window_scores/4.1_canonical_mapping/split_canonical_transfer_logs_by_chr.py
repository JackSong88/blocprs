#!/usr/bin/env python3
"""Annotate pgsc_calc logs with source keys and split them by chromosome."""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import OrderedDict
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import (ANNOTATED_LOG_COLUMNS, normalize_accession,
                                      normalize_chr, parse_chromosomes)


VERBOSE = False


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


# Print progress when verbose output is enabled.
def log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


# Interpret boolean flags in pgsc_calc logs.
def truthy(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"1", "true", "t", "yes"})


class SourceMapCache:

    # Limit the number of source maps held in memory.
    def __init__(self, source_map_root: Path, build: str, max_items: int = 3):
        self.source_map_root = source_map_root
        self.build = build
        self.max_items = max_items
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    # Load a source map and evict the least recently used map if needed.
    def get(self, accession: str) -> pd.DataFrame:
        if accession in self.cache:
            self.cache.move_to_end(accession)
            return self.cache[accession]
        path = self.source_map_root / self.build / f"{accession}.source_map.tsv.gz"
        df = pd.read_csv(path, sep="\t", usecols=SOURCE_COLUMNS, dtype=str, keep_default_na=False)
        df["row_nr"] = pd.to_numeric(df["row_nr"], errors="raise").astype("int64")
        self.cache[accession] = df
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return df


class ChromosomeWriters:

    # Open one labeled output table per chromosome.
    def __init__(self, out_dir: Path, biobank: str, chromosomes: list[str]):
        self.out_dir = out_dir
        self.biobank = biobank
        self.chromosomes = chromosomes
        self.handles = {}
        self.counts = {chrom: 0 for chrom in chromosomes}
        out_dir.mkdir(parents=True, exist_ok=True)
        for chrom in chromosomes:
            path = out_dir / f"{biobank}_chr{chrom}.tsv.gz"
            handle = gzip.open(path, "wt", newline="")
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(ANNOTATED_LOG_COLUMNS)
            self.handles[chrom] = handle

    # Append annotated rows to their chromosome tables.
    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df[ANNOTATED_LOG_COLUMNS]
        for chrom, group in df.groupby("chr_name", sort=False):
            if chrom not in self.handles:
                continue
            group.to_csv(self.handles[chrom], sep="\t", index=False, header=False)
            self.counts[chrom] += len(group)

    # Close all chromosome output files.
    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


# Attach source metadata to retained matches and count annotation mismatches.
def annotate_chunk(chunk: pd.DataFrame, source_cache: SourceMapCache,
                   chromosomes: set[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "input_rows": len(chunk),
        "kept_rows": 0,
        "missing_source_rows": 0,
        "source_mismatch_rows": 0,
    }
    chunk = chunk.copy()
    chunk["chr_name"] = chunk["chr_name"].map(normalize_chr)
    chunk["chr_position"] = pd.to_numeric(chunk["chr_position"], errors="coerce")
    chunk["row_nr"] = pd.to_numeric(chunk["row_nr"], errors="coerce")
    keep = (chunk["chr_name"].isin(chromosomes)
            & chunk["chr_position"].notna() & chunk["row_nr"].notna()
            & chunk["match_status"].fillna("").astype(str).str.lower().eq("matched")
            & truthy(chunk["best_match"]) & ~truthy(chunk["exclude"]))
    chunk = chunk.loc[keep].copy()
    if chunk.empty:
        return pd.DataFrame(columns=ANNOTATED_LOG_COLUMNS), stats

    chunk["accession"] = chunk["accession"].map(normalize_accession)
    chunk["row_nr"] = chunk["row_nr"].astype("int64")
    chunk["chr_position"] = chunk["chr_position"].astype("int64").astype(str)
    chunk = chunk.rename(columns={"ID": "id"}).fillna("")

    annotated = []
    for accession, group in chunk.groupby("accession", sort=False):
        source_map = source_cache.get(accession)
        merged = group.merge(source_map, on="row_nr", how="left", validate="many_to_one")
        missing = merged["source_key"].isna()
        stats["missing_source_rows"] += int(missing.sum())
        if missing.any():
            merged = merged.loc[~missing].copy()
        if merged.empty:
            continue

        mismatch = (merged["effect_allele"].astype(str).ne(merged["source_effect_allele"].astype(str))
                    | merged["other_allele"].astype(str).ne(merged["source_other_allele"].astype(str))
                    | merged["effect_weight"].astype(str).ne(merged["source_effect_weight"].astype(str)))
        stats["source_mismatch_rows"] += int(mismatch.sum())
        merged = merged.loc[~mismatch].copy()
        if merged.empty:
            continue
        annotated.append(merged)

    if not annotated:
        return pd.DataFrame(columns=ANNOTATED_LOG_COLUMNS), stats

    out = pd.concat(annotated, ignore_index=True)
    out["chr_name"] = out["chr_name"].astype(str)
    out["matched_effect_allele"] = out["matched_effect_allele"].astype(str)
    out["effect_type"] = out["effect_type"].astype(str)
    stats["kept_rows"] = len(out)
    return out[ANNOTATED_LOG_COLUMNS], stats


# Stream one cohort's log into chromosome tables and return row counts.
def split_one_log(log_path: Path, biobank: str, build: str, source_map_root: Path,
                  out_dir: Path, chromosomes: list[str], chunksize: int,
                  progress_every: int) -> dict[str, int]:
    source_cache = SourceMapCache(source_map_root, build)
    writers = ChromosomeWriters(out_dir, biobank, chromosomes)
    totals = {
        "input_rows": 0,
        "kept_rows": 0,
        "missing_source_rows": 0,
        "source_mismatch_rows": 0,
    }
    log(f"{biobank}: start log split build={build} log={log_path} "
        f"chunksize={chunksize:,} chromosomes={','.join(chromosomes)}")
    try:
        reader = pd.read_csv(log_path, compression="gzip", usecols=LOG_COLUMNS,
                             dtype=str, chunksize=chunksize, low_memory=False)
        for chunk_idx, chunk in enumerate(reader, start=1):
            annotated, stats = annotate_chunk(chunk, source_cache, set(chromosomes))
            writers.write(annotated)
            for key, value in stats.items():
                totals[key] += value
            if progress_every > 0 and chunk_idx % progress_every == 0:
                log(f"{biobank}: chunk={chunk_idx:,} "
                    f"chunk_input={stats['input_rows']:,} chunk_kept={stats['kept_rows']:,} "
                    f"total_input={totals['input_rows']:,} total_kept={totals['kept_rows']:,} "
                    f"missing_source={totals['missing_source_rows']:,} "
                    f"source_mismatch={totals['source_mismatch_rows']:,}")
    finally:
        writers.close()
    totals.update({f"chr{chrom}_rows": count for chrom, count in writers.counts.items()})
    nonempty = {chrom: count for chrom, count in writers.counts.items() if count}
    log(f"{biobank}: complete total_input={totals['input_rows']:,} "
        f"total_kept={totals['kept_rows']:,} nonempty_chr={len(nonempty)} "
        f"missing_source={totals['missing_source_rows']:,} "
        f"source_mismatch={totals['source_mismatch_rows']:,}")
    for chrom in chromosomes:
        log(f"{biobank}: chr{chrom} annotated_rows={writers.counts[chrom]:,}")
    return totals


# Annotate each cohort using its own build-specific source row map.
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
    global VERBOSE
    VERBOSE = args.verbose
    chromosomes = parse_chromosomes(args.chromosomes)

    if not chromosomes or args.chunksize < 1:
        parser.error("Provide chromosomes and a positive chunk size.")

    for cohort, build, path in (("ukbb", "GRCh37", args.ukbb_log),
                                ("cartagene", "GRCh38", args.cartagene_log)):
        totals = split_one_log(path, cohort, build, args.source_map_root,
                               args.out_dir, chromosomes, args.chunksize, 1)

        if totals["missing_source_rows"] or totals["source_mismatch_rows"]:
            raise ValueError(f"{cohort}: source annotation failed: {totals}")

        print(f"{cohort}: retained {totals['kept_rows']:,} matched source rows.")


if __name__ == "__main__":
    main()
