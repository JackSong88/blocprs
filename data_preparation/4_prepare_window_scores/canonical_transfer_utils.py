#!/usr/bin/env python3
"""Reads PRS weight files and creates keys for matching variant entries across builds."""

from __future__ import annotations

import csv
import gzip
import math
import re
from pathlib import Path
from typing import Iterator


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


# Removes 'chr' from the start of a chromosome label and trims surrounding spaces.
def normalize_chr(value: object) -> str:
    text = str(value).strip()
    return text[3:] if text.lower().startswith("chr") else text


# Removes build labels such as '_hmPOS_GRCh37' from the PRS name.
def normalize_accession(value: object) -> str:
    return BUILD_SUFFIX_RE.sub("", str(value).strip())


# Uses the filename as the PRS name, with the PGS prefix used by pgsc_calc.
def accession_from_source_path(path: Path) -> str:
    stem = path.name

    if stem.endswith(".txt.gz"):
        stem = stem[: -len(".txt.gz")]
    elif stem.endswith(".gz"):
        stem = stem[: -len(".gz")]
    elif stem.endswith(".txt"):
        stem = stem[: -len(".txt")]

    return stem if stem.startswith("PGS") else f"PGS_{stem}"


# Reads the PRS weight file one variant at a time, numbering entries from zero as pgsc_calc does.
def iter_score_rows(path: Path) -> Iterator[tuple[int, dict[str, str], list[str]]]:
    with gzip.open(path, "rt", newline="") as handle:
        header = None

        # Skips '#' metadata lines, then reads the column names.
        for line in handle:
            if not line.startswith("#"):
                header = line.rstrip("\r\n").split("\t")
                break

        if header is None:
            raise RuntimeError(f"No data header found in source scorefile: {path}")

        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")

        for row_nr, row in enumerate(reader):
            yield row_nr, {key: (value or "").strip() for key, value in row.items()}, header


# Reads column names without loading the PRS weights.
def first_score_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                return line.rstrip("\r\n").split("\t")

    raise RuntimeError(f"No data header found in source scorefile: {path}")


# Creates a key from the PRS name, variant ID, alleles and weight.
def make_source_key(accession: str, row: dict[str, str]) -> tuple[str, str, str]:
    effect_allele = row.get("effect_allele", "").strip()
    other_allele = row.get("other_allele", "").strip()
    effect_weight = row.get("effect_weight", "").strip()
    rsid = row.get("rsID", "").strip()

    # Uses the rsID when available, otherwise the original weight file's chromosome and position.
    if rsid not in {"", ".", "NA", "NaN", "nan"}:
        variant_id = rsid
        key_type = "rsid"
    else:
        chrom = normalize_chr(row.get("chr_name", ""))
        pos = row.get("chr_position", "").strip()
        variant_id = f"{chrom}:{pos}"
        key_type = "source_chrpos"

    # Joins the fields into one string, separated by '|'.
    # Includes alleles and weight to distinguish entries for the same variant within a PRS.
    key = "|".join([accession, key_type, variant_id, effect_allele, other_allele, effect_weight])

    return key, key_type, variant_id


# Creates a table row containing the variant's row number, key, coordinates, alleles and weight.
def make_source_map_row(build: str, accession: str, row_nr: int, row: dict[str, str]) -> dict[str, object]:
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


# Expands chromosome ranges such as '1-3,22'; 'all' selects chromosomes 1-22.
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


# Checks whether the weight is nonzero; rejects NaN and infinite values.
def is_nonzero(value: str) -> bool:
    weight = float(value)

    if not math.isfinite(weight):
        raise ValueError(f"Nonfinite score weight: {value}")

    return weight != 0.0
