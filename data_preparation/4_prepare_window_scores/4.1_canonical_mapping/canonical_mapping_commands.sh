#!/bin/bash

set -euo pipefail

source env/blocprs/bin/activate

# Saves variant row numbers, keys, alleles and weights from each original PRS file.
python data_preparation/4_prepare_window_scores/4.1_canonical_mapping/build_canonical_source_maps.py \
    --source-root data/weights \
    --out-root data/canonical/source_maps

# Adds original variant keys, alleles and weights to pgsc_calc's accepted matches.
# Reads 500,000 log entries at a time, using GRCh37 weights for UKB and GRCh38 for CaG.
python data_preparation/4_prepare_window_scores/4.1_canonical_mapping/split_canonical_transfer_logs_by_chr.py \
    --ukbb-log data/ukbb/pgsc/ukbb/match/ukbb_log.csv.gz \
    --cartagene-log data/cartagene/pgsc/cartagene/match/cartagene_log.csv.gz \
    --source-map-root data/canonical/source_maps \
    --out-dir data/canonical/annotated \
    --chromosomes 1-22 \
    --chunksize 500000

# Joins the UKB and CaG tables by source_key, keeping entries matched in both biobanks.
for chromosome in {1..22}; do
    python data_preparation/4_prepare_window_scores/4.1_canonical_mapping/build_canonical_crosswalk_chr.py \
        --chromosome "$chromosome" \
        --logs-dir data/canonical/annotated \
        --out "data/canonical/cross_mapping_chr${chromosome}.tsv.gz"
done

# These crosswalks are the inputs to 4.2_window_scorefiles/assign_canonical_windows_chr.py.
