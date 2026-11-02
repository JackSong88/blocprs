#!/bin/bash

set -euo pipefail

# Uses repository-relative paths and extracted scores; expects a compute allocation.
source env/blocprs/bin/activate
scripts=data_preparation/5_generate_datasets

# Selects panels by AnalysisID to match extraction_commands.sh.
panels=(LDL_MA)

# Available panels from the three tables:
# panels=(
#     HEIGHT_MA LOG_BMI_MA FEV1_FVC_MA LDL_MA LDL_ADJ_MA LOG_HDL_MA TC_MA TC_ADJ_MA
#     LOG_TG_MA DBP_MA DBP_ADJ_MA SBP_MA SBP_ADJ_MA ASTHMA_MA T2D_MA
#     T2D_MT T1D_MT CAD_MT HTN_MT HF_MT AF_MT STR_MT DEM_MT GOUT_MT ASTHMA_MT
#     LOG_TST_SEX URT_SEX LOG_CRTN_SEX WHR_SEX
# )

tables=(
    tables/multi_ancestry_prs_table.csv
    tables/multitrait_prs_table.csv
    tables/sex_biased_prs_table.csv
)

for panel in "${panels[@]}"; do

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

    # Draws one UKB split and reuses it for genome-wide and windowed PRSs.
    # Creates new split files; the dataset commands below read the same files.
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

        # Aligns UKB scores and clinical records to the same saved training/test IDs.
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

        # Keeps all eligible CaG participants for transfer evaluation without a new split.
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

    # Optional check: sum windows and compare against independent genome-wide scores.
    # python "$scripts/check_window_sums.py" \
    #     --windowed "data/datasets/${panel}/ukbb/windowed" \
    #     --genomewide "data/datasets/${panel}/ukbb/genomewide"
done
