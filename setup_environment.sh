#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Creates the environment if it is not already present.
if [[ ! -d env/blocprs ]]; then
    python3 -m venv env/blocprs
fi

# Installs the model, data-preparation and plotting dependencies.
env/blocprs/bin/python -m pip install -e .
