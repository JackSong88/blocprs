#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -d env/blocprs ]]; then
    python3 -m venv env/blocprs
fi

# Install the model and data-preparation dependencies.
env/blocprs/bin/python -m pip install -e '.[preparation]'
