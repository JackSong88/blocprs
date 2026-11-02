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

# Removes temporary downloads when the script exits.
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
  printf 'Downloading sex-specific summary statistics: %s\n' "$phenotype"
  wget --progress=dot:giga -O "$archive" "https://zenodo.org/records/7222725/files/${source_name}.tar.gz?download=1"

  # Extracts the directory and gives the male/female tables consistent filenames.
  tar -xzf "$archive" -C "$temporary"
  mv -T "${temporary}/${source_name}/male_all.${source_name}.glm.linear" "${destination}/M.glm.linear"
  mv -T "${temporary}/${source_name}/female_all.${source_name}.glm.linear" "${destination}/F.glm.linear"
  printf 'Saved male/female summary statistics: %s\n' "$destination"
done
