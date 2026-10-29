#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

out_dir="data/sex_specific_gwas"

phenotypes=("$@")
if [[ "${#phenotypes[@]}" -eq 0 ]]; then
  phenotypes=(URT CRTN TST WHR)
fi
mkdir -p "$out_dir"
temporary=$(mktemp -d "${out_dir}/.sumstats.XXXXXX")
trap 'rm -rf -- "$temporary"' EXIT

for phenotype in "${phenotypes[@]}"; do
  case "$phenotype" in
    URT) source_name=urate ;;
    CRTN) source_name=creatinine ;;
    TST) source_name=testosterone ;;
    WHR) source_name=waist_to_hip ;;
    *) echo "ERROR: unknown phenotype: $phenotype" >&2; exit 1 ;;
  esac
  destination="${out_dir}/${phenotype}"
  mkdir -p "$destination"
  archive="${temporary}/source.tar.gz"
  wget --quiet -O "$archive" "https://zenodo.org/records/7222725/files/${source_name}.tar.gz?download=1"

  tar -xOf "$archive" "${source_name}/male_all.${source_name}.glm.linear" > "${temporary}/M.glm.linear"
  tar -xOf "$archive" "${source_name}/female_all.${source_name}.glm.linear" > "${temporary}/F.glm.linear"
  mv -T "${temporary}/M.glm.linear" "${destination}/M.glm.linear"
  mv -T "${temporary}/F.glm.linear" "${destination}/F.glm.linear"
done
