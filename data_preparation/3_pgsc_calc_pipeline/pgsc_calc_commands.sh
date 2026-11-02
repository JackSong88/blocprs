#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

source env/blocprs/bin/activate
mkdir -p log

# Generates chromosome samplesheets and submits scoring for both cohorts.
for cohort in ukbb cartagene; do
    python data_preparation/3_pgsc_calc_pipeline/prepare_samplesheet.py \
        --sampleset "$cohort"

    sbatch data_preparation/3_pgsc_calc_pipeline/run_pgsc_calc.sh "$cohort"
done
