#!/bin/bash

# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

set -euo pipefail

ld_dir="data/viprs_ld"

mkdir -p "$ld_dir"
archive=$(mktemp "${ld_dir}/.EUR.XXXXXX.tar.gz")
trap 'rm -f -- "$archive"' EXIT

wget --quiet -O "$archive" "https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz"
tar -xzf "$archive" -C "$ld_dir" --no-same-owner --no-same-permissions
