#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

#SBATCH --job-name=blocprs_sex_specific_weights
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=blocprs_sex_specific_weights_%j.log

set -euo pipefail

ld_dir="data/viprs_ld"
sumstats_dir="data/sex_specific_gwas"
output_dir="data/sex_specific_weights"

phenotypes=("$@")
if [[ "${#phenotypes[@]}" -eq 0 ]]; then
  phenotypes=(URT CRTN TST WHR)
fi

mkdir -p "${output_dir}/fits"

for phenotype in "${phenotypes[@]}"; do
  for stratum in M F; do
    viprs_fit -l "${ld_dir}/chr_*/" \
      -s "${sumstats_dir}/${phenotype}/${stratum}.glm.linear" \
      --sumstats-format plink2 \
      --output-dir "${output_dir}/fits" \
      --output-file-prefix "${phenotype}_${stratum}_"

    python3 data_preparation/1_prepare_pgs_weights/utils/harmonize_inferred_beta.py \
      --input-file "${output_dir}/fits/${phenotype}_${stratum}_VIPRS_EM.fit.gz" \
      --pgs-name "${phenotype}_${stratum}" \
      --phenotype "$phenotype" \
      --output-dir "data/weights" \
      --lift-over
  done
done
