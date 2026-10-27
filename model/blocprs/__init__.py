"""Locus-aware integration of fixed source polygenic scores."""

import json

import numpy as np

from .baseline_model import MultiPRS, SingleSourcePRS, genomewide_data, select_source
from .canonical_export import export_canonical_snp_weights
from .data import PRSData, load_dataset
from .export import export_snp_weights, regional_coefficients
from .fixed_composition import FixedCompositionBLocPRS
from .model import BLocPRS

__all__ = ["BLocPRS", "MultiPRS", "SingleSourcePRS", "FixedCompositionBLocPRS",
           "PRSData", "load_dataset", "load_model", "select_source", "genomewide_data",
           "export_snp_weights", "export_canonical_snp_weights", "regional_coefficients"]


# Load the model class recorded in a package archive; never load Python pickles.
def load_model(path):
    with np.load(path, allow_pickle=False) as saved:
        format_name = json.loads(str(saved["metadata"].item()))["format"]

    loaders = {"blocprs-model-v1": BLocPRS,
               "blocprs-baseline-v1": MultiPRS,
               "blocprs-fixed-composition-v1": FixedCompositionBLocPRS}

    if format_name not in loaders:
        raise ValueError(f"Unsupported model archive: {format_name}")

    return loaders[format_name].load(path)
