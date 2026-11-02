#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

# Called by preparation_commands.sh; run on a compute node if used separately.
source env/blocprs/bin/activate

cartagene_dir="$HOME/links/projects/def-sgravel/cartagene"

# Prepares native covariates and cohort cluster assignments.
python data_preparation/2_prepare_validation_data/cartagene/prepare_covariates.py \
    --covariates-file "$cartagene_dir/flagship_GWAS_covariates/GWAS_ALL_covariates.tsv" \
    --clusters-file "$cartagene_dir/clusters/ID_to_clusterID_flagship.tsv" \
    --output-dir data/cartagene/covariates

# Reads medication use before preparing adjusted phenotypes.
python data_preparation/2_prepare_validation_data/cartagene/extract_medication_data.py \
    --metadata-file "$cartagene_dir/old_metadata/data_Gravel936028_2.zip" \
    --output data/cartagene/medication.tsv

# Prepares questionnaire and biochemistry phenotypes.
python data_preparation/2_prepare_validation_data/cartagene/prepare_phenotype_data.py \
    --metadata-file "$cartagene_dir/old_metadata/data_Gravel936028_2.zip" \
    --genetic-codes-file "$cartagene_dir/phenotypes/Gravel936028_5/Gravel_936028_5.genetic_codes.csv.gz" \
    --biochemistry-file "$cartagene_dir/phenotypes/Gravel936028_5/Gravel_936028_5.mesures_biochimiques.csv.gz" \
    --medication-file data/cartagene/medication.tsv \
    --output-dir data/cartagene/phenotypes
