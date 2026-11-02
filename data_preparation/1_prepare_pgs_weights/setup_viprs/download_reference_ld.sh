#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

ld_dir="data/viprs_ld"

mkdir -p "$ld_dir"
archive=$(mktemp "${ld_dir}/.EUR.XXXXXX.tar.gz")
trap 'rm -f -- "$archive"' EXIT

# Downloads and unpacks the European reference LD matrices.
printf 'Downloading European reference LD\n'
wget --progress=dot:giga -O "$archive" "https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz"
printf 'Extracting reference LD to %s\n' "$ld_dir"
tar -xzf "$archive" -C "$ld_dir" --no-same-owner --no-same-permissions
printf 'Reference LD ready: %s\n' "$ld_dir"
