#!/bin/bash

set -euo pipefail

# Processes one chromosome using repository-relative paths; expects a compute allocation.
# Optional arguments select the cohort and chromosome; defaults are UKB and chr1.
source env/blocprs/bin/activate
cohort=${1:-ukbb}
chromosome=${2:-1}

# Score the window weights and the independent whole-chromosome reference.
bash data_preparation/4_prepare_window_scores/4.3_scoring/score_canonical_transfer_windows.sh \
    "$cohort" \
    data/canonical/"$cohort"/"chr${chromosome}"/*.scorefile.gz \
    data/canonical/"${cohort}_fullchr"/"chr${chromosome}"/*.scorefile.gz

# Complete chromosomes 1-22 for both cohorts before extracting their scores.
