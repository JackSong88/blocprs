#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

# Submit from the repository root after all source weights are ready.
source env/blocprs/bin/activate

# Collects source-PRS positions separately for each genome build.
for build in GRCh37 GRCh38; do
    python data_preparation/2_prepare_validation_data/build_variant_ranges.py \
        --weights data/weights/"${build}"/*.txt.gz \
        --genome-build "$build" \
        --output "data/scoring_variants_${build}.tsv"
done

# Prepares cohort tables; the UKB driver submits phenotype and genotype jobs.
bash data_preparation/2_prepare_validation_data/cartagene/cartagene_commands.sh
bash data_preparation/2_prepare_validation_data/ukbb/ukbb_commands.sh

# Wait for the submitted UKB jobs to finish before running pgsc_calc.
