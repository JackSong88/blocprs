#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

# Called by preparation_commands.sh; run on a compute node if used separately.
source env/blocprs/bin/activate

# Prepares eligible participants and native UKB covariates.
python data_preparation/2_prepare_validation_data/ukbb/prepare_samples.py \
    --tabular /project/rpp-aevans-ab/neurohub/UKB/Tabular/current.csv \
    --withdrawals /project/rpp-aevans-ab/neurohub/UKB/Withdrawals/w45551_20250818.csv \
    --keep-out data/ukbb/participants.keep \
    --covariates-out data/ukbb/covariates.tsv

# Reads medication use before preparing adjusted phenotypes.
python data_preparation/2_prepare_validation_data/ukbb/extract_medication_data.py \
    --tabular-file /project/rpp-aevans-ab/neurohub/UKB/Tabular/current.csv \
    --output data/ukbb/medication.tsv

# Required inputs now exist; these two jobs can run independently.
mkdir -p log
sbatch data_preparation/2_prepare_validation_data/ukbb/extract_phenotypes.sh
sbatch data_preparation/2_prepare_validation_data/ukbb/extract_genotypes.sh
