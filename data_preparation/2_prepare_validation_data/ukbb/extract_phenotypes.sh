#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

#SBATCH --job-name=blocprs_ukbb_phenotypes
#SBATCH --account=def-liyue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=log/blocprs_ukbb_phenotypes_%j.log

set -euo pipefail

source env/blocprs/bin/activate

# Prepares continuous phenotypes and medication-adjusted versions.
python data_preparation/2_prepare_validation_data/ukbb/prepare_phenotype_data.py \
  --tabular-file "/project/rpp-aevans-ab/neurohub/UKB/Tabular/current.csv" \
  --withdrawals-file "/project/rpp-aevans-ab/neurohub/UKB/Withdrawals/w45551_20250818.csv" \
  --medication-file "data/ukbb/medication.tsv" \
  --output-dir "data/ukbb/phenotypes"

module load gcc/12.3 r/4.5.0
export R_LIBS="$PWD/env/R_phewas_env"

# Derive binary phenotypes from diagnosis and self-reported codes.
Rscript data_preparation/2_prepare_validation_data/ukbb/extract_binary_phenotypes.R \
  --ukb-file "/project/rpp-aevans-ab/neurohub/UKB/Tabular/current.csv" \
  --withdrawn-file "/project/rpp-aevans-ab/neurohub/UKB/Withdrawals/w45551_20250818.csv" \
  --coding609-file "data/ukbb/raw/coding609.tsv" \
  --phecodes 250.2,250.1,427.2,428,401.1,411,290,274.1,495,433,433.1 \
  --phenotype-names T2D,T1D,AF,HF,HTN,CAD,DEM,GOUT,ASTHMA,STR,STR_SPEC \
  --outdir "data/ukbb/phenotypes" \
  --include-selfreported \
  --apply-phecode-exclusion
