#!/bin/bash

#SBATCH --job-name=blocprs_training
#SBATCH --account=def-liyue
#SBATCH --array=0-28
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=log/blocprs_training_%A_%a.log

set -euo pipefail

# Submit with sbatch from the repository root, with log/ already created.
# Each array index selects the same phenotype as dataset_commands.sh.
datasets=data/datasets
results=data/results
train_ancestry=all
window_mb=1

panels=(
    HEIGHT_MA LOG_BMI_MA FEV1_FVC_MA LDL_MA LDL_ADJ_MA LOG_HDL_MA
    TC_MA TC_ADJ_MA LOG_TG_MA DBP_MA DBP_ADJ_MA SBP_MA SBP_ADJ_MA ASTHMA_MA T2D_MA
    T2D_MT T1D_MT CAD_MT HTN_MT HF_MT AF_MT STR_MT DEM_MT GOUT_MT ASTHMA_MT
    LOG_TST_SEX URT_SEX LOG_CRTN_SEX WHR_SEX
)

while (($#)); do
    case "$1" in
        --datasets) datasets=$2; shift 2 ;;
        --results) results=$2; shift 2 ;;
        --train-ancestry) train_ancestry=$2; shift 2 ;;
        --window-mb) window_mb=$2; shift 2 ;;
        --panels)
            shift
            panels=()
            while (($#)) && [[ "$1" != --* ]]; do
                panels+=("$1")
                shift
            done
            ;;
        --help)
            printf '%s\n' \
                'sbatch model/training_commands.sh [options]' \
                '  --panels HEIGHT_MA URT_SEX   Default: all 29 phenotype panels.' \
                '                              Match --array to this list, e.g. sbatch --array=0-1 ...' \
                '  --train-ancestry EUR         Default: all; also accepts non-EUR or a recorded ancestry.' \
                '  --datasets data/datasets    Dataset root from dataset_commands.sh.' \
                '  --results data/results      Model and evaluation output root.' \
                '  --window-mb 1               Labels the supplied windowed dataset; does not rebuild windows.'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ ${#panels[@]} -gt 0 ]] || { echo 'Select at least one phenotype panel.' >&2; exit 1; }

task=${SLURM_ARRAY_TASK_ID:?Submit with sbatch and select a phenotype array index.}

if ((task < 0 || task >= ${#panels[@]})); then
    echo "Array index $task is outside the ${#panels[@]} selected phenotype panels." >&2
    exit 1
fi

source env/blocprs/bin/activate

# Each CV fold uses one BLAS thread; the final BLocPRS fit uses the allocated CPUs.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

panel=${panels[$task]}
input="$datasets/$panel/ukbb"
output="$results/$panel/train_$train_ancestry/mb$window_mb"
workers=${SLURM_CPUS_PER_TASK:-5}

# Phenotype-panel suffixes identify the source panel, not a different outcome family.
case "${panel%_*}" in
    ASTHMA|T2D|T1D|CAD|HTN|HF|AF|STR|DEM|GOUT) family=binomial ;;
    *) family=gaussian ;;
esac

if [[ -e "$output/multiprs.npz" || -e "$output/blocprs.npz" ]]; then
    echo "Model outputs already exist in $output; choose another --results directory." >&2
    exit 1
fi

# Selects within the saved training split. Excluded participants never become test participants.
ancestry=()
[[ "$train_ancestry" == all ]] || ancestry=(--ancestry "$train_ancestry")

python data_preparation/5_generate_datasets/subset_dataset.py \
    --dataset "$input/windowed" --split train "${ancestry[@]}" --output-dir "$output/training_ids"
ids="$output/training_ids/train.ids.tsv"

# Uses the same training participants and ID order for both models.
printf 'Fitting %s: %s family, training ancestry=%s.\n' "$panel" "$family" "$train_ancestry"
python -u model/fit_baseline.py \
    --dataset "$input/genomewide" --train-ids "$ids" --family "$family" \
    --variant unregularized --seed 42 --output "$output/multiprs.npz"

# Selects the regional penalty from 25 values, 1e-3 to 1e3, using 50,000 total CV participants at most.
python -u model/fit_blocprs.py \
    --dataset "$input/windowed" --train-ids "$ids" --family "$family" \
    --cv-folds 5 --cv-max-samples 50000 --cv-workers "$workers" --fit-workers "$workers" \
    --seed 42 --block-size 8192 --output "$output/blocprs.npz"

printf 'Fitting complete: %s\n' "$output"
