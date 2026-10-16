"""Locus-aware integration of fixed source polygenic scores."""

from .data import PRSData
from .export import export_snp_weights, regional_coefficients
from .model import BLocPRS

__all__ = ["BLocPRS", "PRSData", "export_snp_weights", "regional_coefficients"]
