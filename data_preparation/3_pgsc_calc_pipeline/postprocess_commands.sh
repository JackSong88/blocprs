#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

source env/blocprs/bin/activate

# Run on a compute node after scoring finishes.
# Extracts projected PCs and ancestry assignments for both cohorts.
for cohort in ukbb cartagene; do
    python data_preparation/3_pgsc_calc_pipeline/postprocess_pgsc_data.py \
        --sampleset "$cohort"
done
