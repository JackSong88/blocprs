#!/bin/bash
#SBATCH --job-name=blocprs_ukbb_extract
#SBATCH --array=1-22
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=14:00:00
#SBATCH --output=blocprs_ukbb_extract_%A_%a.log

set -euo pipefail

chrom="${1:-${SLURM_ARRAY_TASK_ID:-1}}"
output_prefix="data/ukbb/genotypes/chr${chrom}"

mkdir -p "$(dirname "$output_prefix")"

plink2 --bgen "data/ukbb/raw/genotypes/chr${chrom}.bgen" ref-first \
  --sample "data/ukbb/raw/genotypes/chr${chrom}.sample" \
  --make-pgen \
  --threads "${SLURM_CPUS_PER_TASK:-4}" \
  --memory 64000 \
  --allow-no-sex \
  --max-alleles 2 \
  --keep "data/ukbb/participants.keep" \
  --extract range "data/scoring_variants_GRCh37.tsv" \
  --out "$output_prefix"
