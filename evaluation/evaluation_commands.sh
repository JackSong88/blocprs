#!/bin/bash

#SBATCH --job-name=blocprs_evaluation
#SBATCH --account=def-liyue
#SBATCH --array=0-28
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=log/blocprs_evaluation_%A_%a.log

set -euo pipefail

# Submit with sbatch after training finishes, from the repository root with log/ already created.
# Each array index selects the same phenotype as dataset_commands.sh and training_commands.sh.
datasets=data/datasets
results=data/results
train_ancestry=all
window_mb=1
cohorts=(ukbb cartagene)
methods=(multiprs blocprs)
requested_groups=()

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
        --cohort) cohorts=("$2"); shift 2 ;;
        --panels|--groups)
            option=$1
            shift
            values=()
            while (($#)) && [[ "$1" != --* ]]; do
                values+=("$1")
                shift
            done
            [[ ${#values[@]} -gt 0 ]] || { echo "$option needs at least one value." >&2; exit 1; }
            if [[ "$option" == --panels ]]; then panels=("${values[@]}"); else requested_groups=("${values[@]}"); fi
            ;;
        --help)
            printf '%s\n' \
                'sbatch evaluation/evaluation_commands.sh [options]' \
                '  --panels HEIGHT_MA URT_SEX   Default: all 29 phenotype panels.' \
                '                              Match --array to this list, e.g. sbatch --array=0-1 ...' \
                '  --train-ancestry EUR         Selects EUR-trained models; default: all-trained models.' \
                '  --groups EUR female         Report only these groups; intersections use EUR:female.' \
                '                              Default: overall, EUR/non-EUR, each ancestry, female/male.' \
                '  --cohort ukbb               Default: both ukbb and cartagene.' \
                '  --datasets data/datasets    Dataset root from dataset_commands.sh.' \
                '  --results data/results      Same output root used for training.' \
                '  --window-mb 1               Same window size supplied to training.'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

for cohort in "${cohorts[@]}"; do
    [[ "$cohort" == ukbb || "$cohort" == cartagene ]] || { echo "Unknown cohort: $cohort" >&2; exit 1; }
done

task=${SLURM_ARRAY_TASK_ID:?Submit with sbatch and select a phenotype array index.}

if ((task < 0 || task >= ${#panels[@]})); then
    echo "Array index $task is outside the ${#panels[@]} selected phenotype panels." >&2
    exit 1
fi

source env/blocprs/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

panel=${panels[$task]}
model_dir="$results/$panel/train_$train_ancestry/mb$window_mb"

for cohort in "${cohorts[@]}"; do

    # These two phenotypes are also omitted from CaG dataset creation.
    if [[ "$cohort" == cartagene && ( "$panel" == DEM_MT || "$panel" == LOG_TST_SEX ) ]]; then
        printf 'Skipping %s in CaG: phenotype unavailable.\n' "$panel"
        continue
    fi

    split=test
    [[ "$cohort" != cartagene ]] || split=all

    if [[ ${#requested_groups[@]} -gt 0 ]]; then
        groups=("${requested_groups[@]}")

        # Commas separate groups; colons still identify intersections such as EUR:female.
        selection=$(IFS=,; printf '%s' "${groups[*]}")
    else

        # Both models share participants; reads their observed ancestries once per cohort.
        group_text=$(python - "$datasets/$panel/$cohort/genomewide/participants.tsv" "$split" <<'PY'
import csv
import sys

with open(sys.argv[1], newline='') as handle:
    ancestries = sorted({row['Ancestry'] for row in csv.DictReader(handle, delimiter='\t')
                         if row['split'] == sys.argv[2] and row['Ancestry']})

groups = ['all'] + (['EUR'] if 'EUR' in ancestries else []) + ['non-EUR']
groups += [ancestry for ancestry in ancestries if ancestry != 'EUR']
groups += ['female', 'male']
print(' '.join(groups))
PY
        )
        read -r -a groups <<< "$group_text"
        selection=stratified
    fi

    # Evaluates the two frozen models sequentially within this phenotype's array task.
    for method in "${methods[@]}"; do
        representation=genomewide
        method_mb=0
        if [[ "$method" == blocprs ]]; then representation=windowed; method_mb=$window_mb; fi
        dataset="$datasets/$panel/$cohort/$representation"
        output="$model_dir/evaluation/$cohort/$selection/$method.tsv"

        if [[ -e "$output" ]]; then
            echo "Evaluation already exists: $output" >&2
            exit 1
        fi

        # Training ancestry selects the saved model; --groups independently selects evaluation participants.
        printf 'Evaluating %s, %s: %s-trained model; cohort=%s; groups=%s.\n' \
            "$panel" "$method" "$train_ancestry" "$cohort" "${groups[*]}"
        python -u evaluation/evaluate.py \
            --dataset "$dataset" --model "$model_dir/$method.npz" --split "$split" \
            --groups "${groups[@]}" --metrics r2 --bootstrap 0 \
            --phenotype "$panel" --cohort "$cohort" --train-group "$train_ancestry" --train-size full \
            --method "$method" --window-mb "$method_mb" --output "$output"

        printf 'Evaluation complete: %s\n' "$output"
    done
done
