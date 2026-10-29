# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Shared phenotype QC and medication corrections used in the cohort scripts."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore


# Flag values more than three population SDs from their stratum mean.
def detect_outliers(phenotype, sigma_threshold=3, stratify=None, nan_policy="omit"):
    if stratify is None:
        stratify = np.ones_like(phenotype)
    mask = np.zeros_like(phenotype, dtype=bool)
    for group in np.unique(stratify):
        selected = stratify == group
        mask[selected] = np.abs(zscore(phenotype[selected], nan_policy=nan_policy)) > sigma_threshold
    return mask


# Screen log components and log ratio, then return FEV1/FVC on its raw scale.
def prepare_fev1_fvc_ratio(fev1, fvc, stratify=None, sigma_threshold=3):
    fev1 = pd.to_numeric(pd.Series(fev1, copy=True).reset_index(drop=True), errors="coerce")
    fvc = pd.to_numeric(pd.Series(fvc, copy=True).reset_index(drop=True), errors="coerce")

    if len(fev1) != len(fvc):
        raise ValueError("FEV1 and FVC must contain the same number of participants.")

    strata = None if stratify is None else pd.Series(stratify, copy=True).reset_index(drop=True)

    if strata is not None and len(strata) != len(fev1):
        raise ValueError("Sex strata must match the number of participants.")

    fev1 = fev1.where(np.isfinite(fev1) & fev1.gt(0))
    fvc = fvc.where(np.isfinite(fvc) & fvc.gt(0))
    fev1 = fev1.mask(detect_outliers(np.log(fev1), sigma_threshold, strata))
    fvc = fvc.mask(detect_outliers(np.log(fvc), sigma_threshold, strata))
    ratio = fev1 / fvc
    ratio = ratio.where(np.isfinite(ratio) & ratio.gt(0))
    return ratio.mask(detect_outliers(np.log(ratio), sigma_threshold, strata)).to_numpy(dtype=float)


def check_unique_ids(frame, column="IID"):
    ids = frame[column]

    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{column} must contain unique, nonmissing participant IDs.")


def read_medication_use(path):
    frame = pd.read_csv(path, sep="\t", dtype={"IID": str, "chol_med": "boolean", "bp_med": "boolean"})
    check_unique_ids(frame)
    frame[["chol_med", "bp_med"]] = frame[["chol_med", "bp_med"]].astype(bool)
    return frame


# Medication correction follows phenotype QC.
def adjust_for_medication(frame, medication, phenotype):
    merged = frame.merge(medication, on="IID", validate="one_to_one")
    if phenotype in {"LDL", "TC"}:

        # GLGC corrections: LDL / 0.7 and total cholesterol / 0.8 for users.
        divisor = 0.7 if phenotype == "LDL" else 0.8
        corrected = merged["phenotype"] / divisor
        use = merged["chol_med"]
    elif phenotype in {"SBP", "DBP"}:

        # Blood-pressure corrections: add 15 mmHg to SBP or 10 mmHg to DBP.
        corrected = merged["phenotype"] + (15 if phenotype == "SBP" else 10)
        use = merged["bp_med"]
    else:
        raise ValueError(f"No medication correction is defined for {phenotype}.")
    merged["phenotype"] = np.where(use, corrected, merged["phenotype"])
    return merged[["FID", "IID", "phenotype"]].copy()


def prepare_output_directory(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_phenotype(frame, output_dir, name):
    frame[["FID", "IID", "phenotype"]].to_csv(Path(output_dir) / f"{name}.txt",
        sep="\t", index=False, header=False, na_rep="NA")
