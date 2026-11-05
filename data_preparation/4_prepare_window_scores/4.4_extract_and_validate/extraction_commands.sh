#!/bin/bash

#SBATCH --job-name=blocprs_extract
#SBATCH --account=def-liyue
#SBATCH --array=0-28
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=15:00:00
#SBATCH --output=log/blocprs_extract_%A_%a.log

set -euo pipefail

# Run from the repository root with log/ already created; defaults to UKB.
# Memory/time are conservative UKB defaults; sbatch options can override them for CaG.
source env/blocprs/bin/activate
cohort=${1:-ukbb}

# Selects one panel per array task, using AnalysisID from the PRS tables.
panels=(
    HEIGHT_MA LOG_BMI_MA FEV1_FVC_MA LDL_MA LDL_ADJ_MA LOG_HDL_MA
    TC_MA TC_ADJ_MA LOG_TG_MA DBP_MA DBP_ADJ_MA SBP_MA SBP_ADJ_MA ASTHMA_MA T2D_MA
    T2D_MT T1D_MT CAD_MT HTN_MT HF_MT AF_MT STR_MT DEM_MT GOUT_MT ASTHMA_MT
    LOG_TST_SEX URT_SEX LOG_CRTN_SEX WHR_SEX
)
panel=${panels[$SLURM_ARRAY_TASK_ID]}

tables=(
    tables/multi_ancestry_prs_table.csv
    tables/multitrait_prs_table.csv
    tables/sex_biased_prs_table.csv
)

printf 'Extracting %s scores for %s.\n' "$panel" "$cohort"

# Reads source IDs from PGS, which also includes the custom sex-specific scores.
source_ids=$(python - "$panel" "${tables[@]}" <<'PY'
import csv
import sys

panel = sys.argv[1]
sources = []

for path in sys.argv[2:]:
    with open(path, newline="") as handle:
        sources.extend(row["PGS"] for row in csv.DictReader(handle) if row["AnalysisID"] == panel)

if not sources:
    raise SystemExit(f"No PRS sources found for panel: {panel}")

print(" ".join(sources))
PY
)
read -r -a sources <<< "$source_ids"
work_dir="${SLURM_TMPDIR:-data/tmp}/extract_${cohort}_${panel}"

# Collects each window's SUM scores; saved blocks allow interrupted runs to resume.
python data_preparation/4_prepare_window_scores/4.4_extract_and_validate/extract_pgsc_scores.py \
    --scorefiles data/canonical/"$cohort"/chr*/*.scorefile.gz \
    --score-dir "data/canonical/${cohort}_scores" \
    --sources "${sources[@]}" \
    --chromosomes 1-22 \
    --block-size 65536 \
    --tmp-dir "data/tmp/extract_scores/${cohort}_${panel}_windowed" \
    --work-dir "$work_dir" \
    --out "data/${cohort}/scores/${panel}/windowed"

# Adds the independently scored chromosomes to obtain genome-wide PRSs.
python data_preparation/4_prepare_window_scores/4.4_extract_and_validate/extract_pgsc_scores.py \
    --scorefiles data/canonical/"${cohort}_fullchr"/chr*/*.scorefile.gz \
    --score-dir "data/canonical/${cohort}_scores" \
    --sources "${sources[@]}" \
    --chromosomes 1-22 \
    --block-size 65536 \
    --tmp-dir "data/tmp/extract_scores/${cohort}_${panel}_genomewide" \
    --work-dir "$work_dir" \
    --out "data/${cohort}/scores/${panel}/genomewide"
