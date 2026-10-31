#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

#SBATCH --job-name=blocprs_pgsc_calc
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=blocprs_pgsc_calc_%j.log

set -euo pipefail

cohort=${1:-ukbb}
case "$cohort" in
  ukbb) build=GRCh37 ;;
  cartagene) build=GRCh38 ;;
  *) echo "Cohort must be ukbb or cartagene." >&2; exit 2 ;;
esac

# Slurm runs a temporary script copy; submit batch jobs from the repository root.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$script_dir/custom.config" ]]; then
  cd -- "$script_dir/../.."
else
  cd -- "$SLURM_SUBMIT_DIR"
fi
root="$PWD"
cohort_dir="$root/data/$cohort"

# Keep pipeline downloads and container images with the project.
export NXF_HOME="$root/data/nextflow"
export NXF_VER=25.10.0
export NXF_SINGULARITY_CACHEDIR="$root/data/containers"
export NXF_OPTS='-Xms1G -Xmx16G'
export APPTAINER_CACHEDIR="$root/data/containers/cache"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

# Use the same temporary directory on the host and inside containers.
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-$cohort_dir/tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export TMPDIR="$APPTAINER_TMPDIR"
export APPTAINERENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+$APPTAINER_BINDPATH,}$TMPDIR"
export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:+$SINGULARITY_BINDPATH,}$TMPDIR"

mkdir -p "$cohort_dir/pgsc" "$cohort_dir/pgsc_work" "$cohort_dir/pgsc_launch" \
  "$NXF_HOME" "$NXF_SINGULARITY_CACHEDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Separate Nextflow logs and retained work for each cohort.
cd "$cohort_dir/pgsc_launch"
"$NXF_HOME/nextflow" -c "$root/data_preparation/3_pgsc_calc_pipeline/custom.config" run pgscatalog/pgsc_calc \
  -r v2.1.0 \
  -profile singularity \
  --custom_config_base "$root/data/nextflow/offline_configs" \
  -work-dir "$cohort_dir/pgsc_work" \
  --input "$cohort_dir/samplesheet.csv" \
  --target_build "$build" \
  --scorefile "$root/data/weights/$build/*.txt.gz" \
  --outdir "$cohort_dir/pgsc" \
  --min_overlap 0.75 \
  --run_ancestry "$root/data/reference/pgsc_HGDP+1kGP_v1.tar.zst"
