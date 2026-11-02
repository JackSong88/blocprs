#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

#SBATCH --job-name=blocprs_ukbb_extract
#SBATCH --account=def-liyue
#SBATCH --array=1-22
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=14:00:00
#SBATCH --output=log/blocprs_ukbb_extract_%A_%a.log

set -euo pipefail

module load plink

chrom="${1:-${SLURM_ARRAY_TASK_ID:-1}}"
output_prefix="${2:-data/ukbb/genotypes/chr${chrom}}"

mkdir -p "$(dirname "$output_prefix")"

# Extracts the selected participants and source-PRS variant positions.
plink2 --bgen "/project/rpp-aevans-ab/neurohub/UKB/Bulk/Imputation/UKB_imputation_from_genotype/ukb22828_c${chrom}_b0_v3.bgen" ref-first \
  --sample "/project/rpp-aevans-ab/neurohub/UKB/Bulk/Imputation/UKB_imputation_from_genotype/ukb22828_c${chrom}_b0_v3.sample" \
  --make-pgen \
  --threads "${SLURM_CPUS_PER_TASK:-4}" \
  --memory "${PLINK_MEMORY_MB:-64000}" \
  --allow-no-sex \
  --max-alleles 2 \
  --keep "data/ukbb/participants.keep" \
  --extract range "data/scoring_variants_GRCh37.tsv" \
  --out "$output_prefix"
