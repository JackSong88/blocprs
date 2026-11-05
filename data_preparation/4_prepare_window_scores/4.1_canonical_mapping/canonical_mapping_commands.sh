#!/bin/bash

#SBATCH --job-name=canonical_mapping
#SBATCH --account=def-liyue
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=log/canonical_mapping_%j.log

set -euo pipefail

source env/blocprs/bin/activate

# Saves a lookup table linking original PRS row numbers to variant details and matching keys.
# The next script uses the row numbers reported by pgsc_calc to retrieve these details.
python data_preparation/4_prepare_window_scores/4.1_canonical_mapping/build_canonical_source_maps.py \
    --source-root data/weights \
    --out-root data/canonical_mapping/intermediate/source_maps

# Extracts pgsc_calc's accepted matches and writes one annotated log per cohort and chromosome.
# Uses the original row numbers to attach each variant's matching key, original coordinates,
# alleles and weight, alongside the genotype ID and effect allele matched by pgsc_calc.
python data_preparation/4_prepare_window_scores/4.1_canonical_mapping/split_canonical_transfer_logs_by_chr.py \
    --ukbb-log data/ukbb/pgsc/ukbb/match/ukbb_log.csv.gz \
    --cartagene-log data/cartagene/pgsc/cartagene/match/cartagene_log.csv.gz \
    --source-map-root data/canonical_mapping/intermediate/source_maps \
    --out-dir data/canonical_mapping/intermediate/annotated_logs \
    --chromosomes 1-22 \
    --chunksize 500000

# Creates the shared UKB/CaG crosswalk using the original PRS entry's matching key.
# Key: PRS name, variant ID (or original coordinates), original alleles and weight.
# Value: shared weight, plus each cohort's genotype ID, matched effect allele and coordinates.
# Keeps only keys matched in both cohorts.
for chromosome in {1..22}; do
    python data_preparation/4_prepare_window_scores/4.1_canonical_mapping/build_canonical_crosswalk_chr.py \
        --chromosome "$chromosome" \
        --logs-dir data/canonical_mapping/intermediate/annotated_logs \
        --out "data/canonical_mapping/crosswalks/cross_mapping_chr${chromosome}.tsv.gz"
done

# These crosswalks are the inputs to 4.2_window_scorefiles/assign_canonical_windows_chr.py.
