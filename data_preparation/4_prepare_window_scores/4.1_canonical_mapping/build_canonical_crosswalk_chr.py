"""Joins UKB and CaG matches by their original PRS variant keys.

Keeps only entries matched in both biobanks. Saves the original weight alongside
each biobank's genotype ID, matched effect allele and coordinates.
Uses pgsc_calc's allele matches.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import CROSSWALK_COLUMNS, normalize_chr


# Reads a chromosome table and adds the biobank name to its genotype column names.
def read_annotated(path, cohort):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame["chr_name"] = frame["chr_name"].map(normalize_chr)

    return frame.rename(columns={"chr_name": f"{cohort}_chr",
                                 "chr_position": f"{cohort}_position",
                                 "id": f"{cohort}_id",
                                 "matched_effect_allele": f"{cohort}_effect_allele"})


# Joins UKB and CaG tables by source_key, keeping entries present in both.
def build_crosswalk(left, right):
    right_columns = ["source_key", "cartagene_id", "cartagene_effect_allele",
                     "cartagene_chr", "cartagene_position"]

    # Joins UKB and CaG using the original PRS key, not their genotype IDs or positions.
    # Keeps shared keys and stops if either table contains duplicate keys.
    merged = left.merge(right[right_columns], on="source_key", how="inner", validate="one_to_one")

    # Each key links to the weight and both cohorts' genotype IDs, matched
    # effect alleles and coordinates. These are separate columns in the table.
    # Illustrative example, not real variant data:
    #   key: PGS000001|rsid|rs123|A|G|0.02
    #   weight: 0.02
    #   UKB: id=variant_U, effect_allele=A, chr=1, position=12345
    #   CaG: id=variant_C, effect_allele=A, chr=1, position=12567

    if merged.empty:
        raise ValueError("No shared matched source variants on this chromosome.")

    # Copies the original PRS weight and selects the columns to save.
    merged["effect_weight"] = merged["source_effect_weight"]
    result = merged[CROSSWALK_COLUMNS].copy()

    for cohort in ("ukbb", "cartagene"):
        result[f"{cohort}_position"] = pd.to_numeric(result[f"{cohort}_position"], errors="raise").astype("int64")

    return result.sort_values(["ukbb_chr", "ukbb_position", "accession", "source_key"])


# Reads and joins one chromosome's UKB and CaG tables, then saves the result.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Reads the accepted matches from each cohort.
    chromosome = normalize_chr(args.chromosome)
    print(f"chr{chromosome}: reading UKB/CaG matches and joining original PRS entries.", flush=True)
    left = read_annotated(args.logs_dir / f"ukbb_chr{chromosome}.tsv.gz", "ukbb")
    right = read_annotated(args.logs_dir / f"cartagene_chr{chromosome}.tsv.gz", "cartagene")

    # Saves only source variants matched in both cohorts.
    result = build_crosswalk(left, right)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, sep="\t", index=False)
    print(f"chr{chromosome}: {len(result):,} shared source variants -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
