"""Intersect eligible source variants between UKB and CARTaGENE, one chromosome at a time."""

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import CROSSWALK_COLUMNS, normalize_chr


# Keep each cohort's genotype ID and matched allele alongside the source key.
def read_annotated(path, cohort):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame["chr_name"] = frame["chr_name"].map(normalize_chr)
    return frame.rename(columns={"chr_name": f"{cohort}_chr",
                                 "chr_position": f"{cohort}_position",
                                 "id": f"{cohort}_id",
                                 "matched_effect_allele": f"{cohort}_effect_allele"})


# Each expert keeps its own shared variants; experts need not share a SNP panel.
def build_crosswalk(left, right):
    right_columns = ["source_key", "cartagene_id", "cartagene_effect_allele",
                     "cartagene_chr", "cartagene_position"]
    merged = left.merge(right[right_columns], on="source_key", how="inner", validate="one_to_one")

    if merged.empty:
        raise ValueError("No shared matched source variants on this chromosome.")

    merged["effect_weight"] = merged["source_effect_weight"]
    result = merged[CROSSWALK_COLUMNS].copy()
    for cohort in ("ukbb", "cartagene"):
        result[f"{cohort}_position"] = pd.to_numeric(result[f"{cohort}_position"], errors="raise").astype("int64")
    return result.sort_values(["ukbb_chr", "ukbb_position", "accession", "source_key"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chromosome = normalize_chr(args.chromosome)
    left = read_annotated(args.logs_dir / f"ukbb_chr{chromosome}.tsv.gz", "ukbb")
    right = read_annotated(args.logs_dir / f"cartagene_chr{chromosome}.tsv.gz", "cartagene")
    result = build_crosswalk(left, right)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, sep="\t", index=False)
    print(f"chr{chromosome}: {len(result):,} shared source variants -> {args.out}")


if __name__ == "__main__":
    main()
