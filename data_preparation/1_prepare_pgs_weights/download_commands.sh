#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

# Uses paths relative to the repository root.
source env/blocprs/bin/activate

# Download PGS Catalog scores in both genome builds.
python3 data_preparation/1_prepare_pgs_weights/download_weights.py \
    --pgs-table tables/multi_ancestry_prs_table.csv \
    --builds GRCh37 GRCh38
python3 data_preparation/1_prepare_pgs_weights/download_weights.py \
    --pgs-table tables/multitrait_prs_table.csv \
    --builds GRCh37 GRCh38

# Downloads sex-specific GWAS summary statistics and reference LD.

# sumstats
bash data_preparation/1_prepare_pgs_weights/prepare_sex_biased_pgs/download_sumstats.sh

# LD
bash data_preparation/1_prepare_pgs_weights/setup_viprs/download_reference_ld.sh

# Runs VIPRS on the sex-stratified phenotypes
mkdir -p log
sbatch data_preparation/1_prepare_pgs_weights/prepare_sex_biased_pgs/train_viprs.sh
