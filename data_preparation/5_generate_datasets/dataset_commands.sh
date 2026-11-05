#!/bin/bash

#SBATCH --job-name=blocprs_datasets
#SBATCH --account=def-liyue
#SBATCH --array=0-28
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=log/blocprs_datasets_%A_%a.log

set -euo pipefail

# Run from the repository root with log/ already created.
# Creates one shared UKB split per panel; CaG keeps all eligible participants.
source env/blocprs/bin/activate
scripts=data_preparation/5_generate_datasets

# Selects one panel per array task, matching extraction_commands.sh.
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

printf 'Creating genome-wide and windowed datasets for %s.\n' "$panel"

# Reads the exact phenotype filename from the tables, including adjusted names.
phenotype=$(python - "$panel" "${tables[@]}" <<'PY'
import csv
import sys

panel = sys.argv[1]
phenotypes = set()

for table in sys.argv[2:]:
    with open(table, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["AnalysisID"] == panel:
                phenotypes.add(row["Phenotype_short"])

if len(phenotypes) != 1 or "" in phenotypes:
    raise SystemExit(f"{panel}: expected one Phenotype_short; found {sorted(phenotypes)}")

print(phenotypes.pop())
PY
)

# Identifies binary phenotypes for case/control-stratified splitting.
case "$phenotype" in
    ASTHMA|T2D|T1D|CAD|HTN|HF|AF|STR|DEM|GOUT) family=binomial ;;
    *) family=gaussian ;;
esac

# Checks that both score inputs cover the same eligible UKB participants.
# Draws one split; both datasets below reuse its training and test ID files.
python "$scripts/create_splits.py" \
    --scores-files "data/ukbb/scores/${panel}/genomewide" "data/ukbb/scores/${panel}/windowed" \
    --phenotype-file "data/ukbb/phenotypes/${phenotype}.txt" \
    --covariates-file data/ukbb/projected_covariates.tsv \
    --ancestry-file data/ukbb/ancestry.tsv \
    --family "$family" \
    --num-pcs 10 \
    --test-fraction 0.3 \
    --seed 7209 \
    --output-dir "data/splits/${panel}"

for scores in genomewide windowed; do

    # Writes scores, phenotype values and covariates in the same saved train/test order.
    python "$scripts/create_datasets.py" \
        --scores-file "data/ukbb/scores/${panel}/${scores}" \
        --phenotype-file "data/ukbb/phenotypes/${phenotype}.txt" \
        --covariates-file data/ukbb/projected_covariates.tsv \
        --ancestry-file data/ukbb/ancestry.tsv \
        --family "$family" \
        --num-pcs 10 \
        --train-ids-file "data/splits/${panel}/train_ids.txt" \
        --test-ids-file "data/splits/${panel}/test_ids.txt" \
        --output-dir "data/datasets/${panel}/ukbb/${scores}"

    # DEM and LOG_TST are unavailable in the prepared CaG phenotypes.
    case "$phenotype" in
        DEM|LOG_TST)
            printf 'CaG %s/%s omitted: phenotype %s is unavailable.\n' "$panel" "$scores" "$phenotype"
            continue
            ;;
    esac

    # Matches CaG scores to phenotype/covariate records by ID and labels every row as all.
    python "$scripts/create_datasets.py" \
        --scores-file "data/cartagene/scores/${panel}/${scores}" \
        --phenotype-file "data/cartagene/phenotypes/${phenotype}.txt" \
        --covariates-file data/cartagene/projected_covariates.tsv \
        --ancestry-file data/cartagene/ancestry.tsv \
        --family "$family" \
        --num-pcs 10 \
        --all-participants \
        --output-dir "data/datasets/${panel}/cartagene/${scores}"
done

# Optional after dataset creation: matches participant/source labels, sums windows,
# and compares with independently scored genome-wide PRSs within rounding tolerances.
# python "$scripts/check_window_sums.py" \
#     --windowed "data/datasets/${panel}/ukbb/windowed" \
#     --genomewide "data/datasets/${panel}/ukbb/genomewide"

printf 'Datasets complete: %s.\n' "$panel"
