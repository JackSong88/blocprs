#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

#SBATCH --job-name=blocprs_ukbb_phenotypes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=blocprs_ukbb_phenotypes_%j.log

set -euo pipefail

python data_preparation/2_prepare_validation_data/ukbb/prepare_phenotype_data.py \
  --tabular-file "data/ukbb/raw/tabular.csv" \
  --withdrawals-file "data/ukbb/raw/withdrawals.csv" \
  --medication-file "data/ukbb/medication.tsv" \
  --output-dir "data/ukbb/phenotypes"

Rscript data_preparation/2_prepare_validation_data/ukbb/extract_binary_phenotypes.R \
  --ukb-file "data/ukbb/raw/tabular.csv" \
  --withdrawn-file "data/ukbb/raw/withdrawals.csv" \
  --coding609-file "data/ukbb/raw/coding609.tsv" \
  --phecodes 250.2,250.1,427.2,428,401.1,411,290,274.1,495,433,433.1 \
  --phenotype-names T2D,T1D,AF,HF,HTN,CAD,DEM,GOUT,ASTHMA,STR,STR_SPEC \
  --outdir "data/ukbb/phenotypes" \
  --include-selfreported \
  --apply-phecode-exclusion
