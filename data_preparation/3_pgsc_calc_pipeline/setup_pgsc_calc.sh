#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
root="$PWD"

export NXF_HOME="$root/data/nextflow"
export NXF_VER=25.10.0
export NXF_SINGULARITY_CACHEDIR="$root/data/containers"
export APPTAINER_CACHEDIR="$root/data/containers/cache"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
unset NXF_OFFLINE

mkdir -p "$NXF_HOME/offline_configs" "$NXF_SINGULARITY_CACHEDIR" \
    "$APPTAINER_CACHEDIR" "$root/data/reference"

# Finish downloads before exposing their final filenames.
download() {
    local url="$1" output="$2"

    if [[ ! -f "$output" ]]; then
        curl --fail --location --retry 3 "$url" --output "$output.part"
        mv -- "$output.part" "$output"
    fi
}

# Cache the same framework, pipeline and plugins used by the scoring runner.
download "https://github.com/nextflow-io/nextflow/releases/download/v${NXF_VER}/nextflow" \
    "$NXF_HOME/nextflow"
chmod +x "$NXF_HOME/nextflow"
"$NXF_HOME/nextflow" -version
"$NXF_HOME/nextflow" pull pgscatalog/pgsc_calc -r v2.1.0
"$NXF_HOME/nextflow" plugin install nf-schema@2.0.0
"$NXF_HOME/nextflow" plugin install nf-prov@1.2.2

# The local executor does not need remote institutional profiles.
if [[ ! -f "$NXF_HOME/offline_configs/nfcore_custom.config" ]]; then
    touch "$NXF_HOME/offline_configs/nfcore_custom.config"
fi

# Containers declared by pgsc_calc v2.1.0.
images=(
    pygscatalog:pgscatalog-utils-2.0.0-singularity
    plink2:2.00a5.10-singularity
    zstd:2-beta-singularity
    report:2-beta-singularity
    pyyaml:6.0-singularity
    fraposa_pgsc:v1.0.2-singularity
)

for image in "${images[@]}"; do
    image="oras://ghcr.io/pgscatalog/$image"

    # Match Nextflow's Singularity cache filenames.
    name=${image#*://}
    name=${name//\//-}
    name=${name//:/-}
    output="$NXF_SINGULARITY_CACHEDIR/$name.img"

    if [[ ! -f "$output" ]]; then
        apptainer pull --force "$output.part" "$image"
        mv -- "$output.part" "$output"
    fi
done

download https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst \
    "$root/data/reference/pgsc_HGDP+1kGP_v1.tar.zst"

for builds in hg19ToHg38 hg38ToHg19; do
    source_build=${builds%%To*}
    download "https://hgdownload.cse.ucsc.edu/goldenpath/$source_build/liftOver/$builds.over.chain.gz" \
        "$root/data/reference/$builds.over.chain.gz"
done
