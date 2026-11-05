#!/bin/bash

set -euo pipefail

# Submits one chromosome; run this launcher from the repository root.
# Optional arguments select the cohort and chromosome; defaults are UKB and chr1.
source env/blocprs/bin/activate
cohort=${1:-ukbb}
chromosome=${2:-1}

mkdir -p log

# Submits windowed scoring and the independent whole-chromosome reference in one job.
sbatch data_preparation/4_prepare_window_scores/4.3_scoring/score_canonical_transfer_windows.sh \
    "$cohort" \
    data/canonical/"$cohort"/"chr${chromosome}"/*.scorefile.gz \
    data/canonical/"${cohort}_fullchr"/"chr${chromosome}"/*.scorefile.gz

# Complete chromosomes 1-22 for both cohorts before extracting their scores.
