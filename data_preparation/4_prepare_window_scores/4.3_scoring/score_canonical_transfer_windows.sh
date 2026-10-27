#!/bin/bash

#SBATCH --job-name=blocprs_score
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=blocprs_score_%j.log

set -euo pipefail

cohort=${1:?Specify UKB or CaG, followed by the chromosome scorefiles.}
shift

if [[ $# -eq 0 ]]; then
  echo "No scorefiles supplied." >&2
  exit 1
fi

case "$cohort" in
  UKB|ukbb)
    sampleset=ukbb
    genotype_dir="data/ukbb/genotypes"
    frequencies="data/ukbb/pgsc_work/ancestry/relabel/afreq/ukbb.afreq_ALL_relabelled.gz"
    max_allele_length=100
    ;;
  CaG|cartagene)
    sampleset=cartagene
    genotype_dir="data/cartagene/genotypes"
    frequencies="data/cartagene/pgsc_work/ancestry/relabel/afreq/cartagene.afreq_ALL_relabelled.gz"
    max_allele_length=1000
    ;;
  *) echo "Cohort must be UKB or CaG." >&2; exit 2 ;;
esac

output="data/canonical/${sampleset}_scores"
mkdir -p "$output"

for score in "$@"; do
  name=$(basename "$score" .scorefile.gz)

  if [[ ! "$name" =~ ^(ukbb|cartagene)_([0-9]+)_additive_([0-9]+)_(w[0-9]+|fullchr)$ ]]; then
    echo "Unexpected scorefile name: $name" >&2
    exit 1
  fi

  chrom=${BASH_REMATCH[2]}

  [[ "${BASH_REMATCH[1]}" == "$sampleset" ]] || { echo "Scorefile cohort differs from $cohort." >&2; exit 1; }

  prefix="$output/$name"
  n_columns=$(python - "$score" <<'PY'
import gzip
import sys
with gzip.open(sys.argv[1], "rt") as handle:
    print(len(handle.readline().rstrip("\n").split("\t")))
PY
)

  [[ "$n_columns" -ge 3 ]] || { echo "No source weights in $score" >&2; exit 1; }

  # SUM, not AVG: scores must add across windows.
  plink2 --pfile "${genotype_dir}/chr${chrom}" \
    --threads "${SLURM_CPUS_PER_TASK:-4}" --memory "${PLINK_MEMORY_MB:-36000}" \
    --seed 31 --allow-extra-chr \
    --set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len "$max_allele_length" missing \
    --read-freq "$frequencies" --chr "$chrom" --extract "$score" \
    --score "$score" 1 2 zs header-read cols=+scoresums,+denom,+fid list-variants \
    --score-col-nums "3-$n_columns" --out "$prefix"
done
