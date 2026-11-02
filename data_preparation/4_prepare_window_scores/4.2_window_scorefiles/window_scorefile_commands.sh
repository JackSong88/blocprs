#!/bin/bash

set -euo pipefail

# Uses repository-relative paths and canonical mapping outputs; expects a compute allocation.
source env/blocprs/bin/activate
scripts=data_preparation/4_prepare_window_scores/4.2_window_scorefiles

# Defines 1-Mb windows using the UKB genotype positions for both biobanks.
python "$scripts/make_windows_from_bim.py" \
    --pvar-template 'data/ukbb/genotypes/chr{chr}.pvar' \
    --chromosomes 1-22 \
    --mb-size 1000000 \
    --out tables/windows_GRCh37_1Mb.tsv

mkdir -p data/canonical/assigned

for chromosome in {1..22}; do

    # Adds the same window assignment to each variant's UKB and CaG matches.
    # Writes the assigned tables used later by SNP-weight export.
    python "$scripts/assign_canonical_windows_chr.py" \
        --chromosome "$chromosome" \
        --shared-crosswalk "data/canonical/cross_mapping_chr${chromosome}.tsv.gz" \
        --windows tables/windows_GRCh37_1Mb.tsv \
        --out "data/canonical/assigned/cross_mapping_chr${chromosome}.tsv.gz"

    for cohort in ukbb cartagene; do

        # pgsc_calc writes matched weights into files named additive_0, additive_1, etc.
        # Includes all additive files, since several can contribute to the same PRS/window.

        # Writes per-window weights using each biobank's matched variant IDs and alleles.
        python "$scripts/split_canonical_scorefiles_chr.py" \
            --biobank "$cohort" \
            --crosswalk "data/canonical/assigned/cross_mapping_chr${chromosome}.tsv.gz" \
            --scorefiles data/"$cohort"/pgsc/"$cohort"/match/"${cohort}_${chromosome}"_additive_*.scorefile.gz \
            --out-dir "data/canonical/${cohort}/chr${chromosome}"

        # Writes whole-chromosome weights using exactly the same retained variants.
        python "$scripts/split_canonical_scorefiles_chr.py" \
            --biobank "$cohort" \
            --crosswalk "data/canonical/assigned/cross_mapping_chr${chromosome}.tsv.gz" \
            --scorefiles data/"$cohort"/pgsc/"$cohort"/match/"${cohort}_${chromosome}"_additive_*.scorefile.gz \
            --out-dir "data/canonical/${cohort}_fullchr/chr${chromosome}" \
            --full-chromosome
    done
done
