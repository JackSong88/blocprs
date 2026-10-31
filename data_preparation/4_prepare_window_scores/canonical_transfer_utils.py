#!/usr/bin/env python3
"""Shared helpers for canonical UKBB -> CARTaGENE transfer scripts."""

from __future__ import annotations

import csv
import gzip
import math
import re
from pathlib import Path
from typing import Iterable, Iterator


BUILD_SUFFIX_RE = re.compile(r"_hmPOS_GRCh(?:37|38)$")


SOURCE_MAP_COLUMNS = [
    "build",
    "accession",
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


ANNOTATED_LOG_COLUMNS = [
    "accession",
    "source_key",
    "source_key_type",
    "source_variant_id",
    "source_chr",
    "source_position",
    "source_effect_allele",
    "source_other_allele",
    "source_effect_weight",
    "row_nr",
    "chr_name",
    "chr_position",
    "id",
    "matched_effect_allele",
    "effect_allele",
    "other_allele",
    "effect_weight",
    "effect_type",
]


CROSSWALK_COLUMNS = [
    "accession",
    "source_key",
    "source_key_type",
    "source_variant_id",
    "source_chr",
    "source_position",
    "source_effect_allele",
    "source_other_allele",
    "effect_weight",
    "ukbb_id",
    "ukbb_effect_allele",
    "ukbb_chr",
    "ukbb_position",
    "cartagene_id",
    "cartagene_effect_allele",
    "cartagene_chr",
    "cartagene_position",
]


# Remove the chromosome prefix and surrounding whitespace.
def normalize_chr(value: object) -> str:
    text = str(value).strip()
    return text[3:] if text.lower().startswith("chr") else text


# Remove the harmonized-build suffix from a source name.
def normalize_accession(value: object) -> str:
    return BUILD_SUFFIX_RE.sub("", str(value).strip())


# Derive the source name from its scorefile name.
def accession_from_source_path(path: Path) -> str:
    stem = path.name
    if stem.endswith(".txt.gz"):
        stem = stem[: -len(".txt.gz")]
    elif stem.endswith(".gz"):
        stem = stem[: -len(".gz")]
    elif stem.endswith(".txt"):
        stem = stem[: -len(".txt")]
    return stem if stem.startswith("PGS") else f"PGS_{stem}"


# Locate the build-specific scorefile for a source.
def source_file_for_accession(source_root: Path, build: str, accession: str) -> Path:
    candidates = [f"{accession}.txt.gz"]
    if accession.startswith("PGS_"):
        candidates.append(f"{accession[4:]}.txt.gz")
    for name in candidates:
        path = source_root / build / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No source scorefile found for accession={accession} build={build} under {source_root}")


# Open a plain or gzip-compressed text file.
def open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def iter_score_rows(path: Path) -> Iterator[tuple[int, dict[str, str], list[str]]]:
    """Yield 0-based source row number plus row dict from a PGS scoring file."""
    with gzip.open(path, "rt", newline="") as handle:
        header = None
        for line in handle:
            if not line.startswith("#"):
                header = line.rstrip("\r\n").split("\t")
                break

        if header is None:
            raise RuntimeError(f"No data header found in source scorefile: {path}")

        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")
        for row_nr, row in enumerate(reader):
            yield row_nr, {key: (value or "").strip() for key, value in row.items()}, header


# Read the scorefile columns, including files with no data rows.
def first_score_header(path: Path) -> list[str]:
    for _, _, header in iter_score_rows(path):
        return header
    with gzip.open(path, "rt", newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                return line.rstrip("\r\n").split("\t")
    raise RuntimeError(f"No data header found in source scorefile: {path}")


# Check whether the source row supplies an rsID.
def has_nonempty_rsid(row: dict[str, str]) -> bool:
    value = row.get("rsID", "").strip()
    return value not in {"", ".", "NA", "NaN", "nan"}


def make_source_key(accession: str, row: dict[str, str]) -> tuple[str, str, str]:
    """Return source_key, key_type, source_variant_id for one source score row."""
    effect_allele = row.get("effect_allele", "").strip()
    other_allele = row.get("other_allele", "").strip()
    effect_weight = row.get("effect_weight", "").strip()
    if has_nonempty_rsid(row):
        variant_id = row["rsID"].strip()
        key_type = "rsid"
    else:
        chrom = normalize_chr(row.get("chr_name", ""))
        pos = row.get("chr_position", "").strip()
        variant_id = f"{chrom}:{pos}"
        key_type = "source_chrpos"

    # Match original source effects across builds, not cohort-local variant IDs.
    # Alleles and weight distinguish different entries for the same SNP.
    key = "|".join([accession, key_type, variant_id, effect_allele, other_allele, effect_weight])
    return key, key_type, variant_id


# Record the source key, coordinates, alleles and weight for one row.
def make_source_map_row(build: str, accession: str, row_nr: int, row: dict[str, str],
                        source_file: Path) -> dict[str, object]:
    source_key, key_type, variant_id = make_source_key(accession, row)
    return {
        "build": build,
        "accession": accession,
        "row_nr": row_nr,
        "source_key": source_key,
        "source_key_type": key_type,
        "source_variant_id": variant_id,
        "source_chr": normalize_chr(row.get("chr_name", "")),
        "source_position": row.get("chr_position", "").strip(),
        "source_effect_allele": row.get("effect_allele", "").strip(),
        "source_other_allele": row.get("other_allele", "").strip(),
        "source_effect_weight": row.get("effect_weight", "").strip(),
    }


# Expand a chromosome list, ranges or all autosomes.
def parse_chromosomes(spec: str) -> list[str]:
    if spec == "all":
        return [str(i) for i in range(1, 23)]
    chroms: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chroms.extend(str(i) for i in range(int(start), int(end) + 1))
        else:
            chroms.append(str(int(token)))
    return chroms


# Write a labeled table and return the number of data rows.
def write_tsv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


# Reject malformed weights; zero is reserved for genuine absent contributions.
def is_nonzero(value: str) -> bool:
    weight = float(value)

    if not math.isfinite(weight):
        raise ValueError(f"Nonfinite score weight: {value}")

    return weight != 0.0
