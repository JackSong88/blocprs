# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Prepare the UKB continuous phenotypes used in the paper.

Sex-stratified outlier screening is applied to the eligible cohort before
dataset splitting. No PRS or fitted model is used.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (adjust_for_medication, check_unique_ids, detect_outliers,
                   prepare_fev1_fvc_ratio, prepare_output_directory, read_medication_use,
                   write_phenotype)

PHENOTYPES = {"50-0.0": "HEIGHT", "21001-0.0": "LOG_BMI", "30760-0.0": "LOG_HDL",
              "30780-0.0": "LDL", "30690-0.0": "TC", "30870-0.0": "LOG_TG",
              "30700-0.0": "LOG_CRTN", "30880-0.0": "URT", "30850-0.0": "LOG_TST",
              "4080-0.0": "SBP", "4079-0.0": "DBP"}
COMPONENTS = ["48-0.0", "49-0.0", "20150-0.0", "20151-0.0"]
ADJUSTED = {"LDL", "TC", "SBP", "DBP"}


def prepare_phenotypes(tabular_file, withdrawals_file, medication_file, output_dir):
    columns = ["eid", "22001-0.0"] + list(PHENOTYPES) + COMPONENTS
    frame = pd.read_csv(tabular_file, usecols=columns, dtype={"eid": str})
    frame = frame.rename(columns={"eid": "IID", "22001-0.0": "Sex"})
    check_unique_ids(frame)
    withdrawn = pd.read_csv(withdrawals_file, names=["IID"], dtype=str)
    frame = frame.loc[~frame["IID"].isin(withdrawn["IID"])].copy()
    frame["FID"] = frame["IID"]
    medication = read_medication_use(medication_file)
    names = list(PHENOTYPES.values()) + [f"{name}_adj" for name in sorted(ADJUSTED)] + ["FEV1_FVC", "WHR"]
    output_dir = prepare_output_directory(output_dir)

    for field, name in PHENOTYPES.items():
        phenotype = frame[["FID", "IID", field]].rename(columns={field: "phenotype"}).copy()
        if name.startswith("LOG_"):
            phenotype["phenotype"] = np.log(phenotype["phenotype"])
        outliers = detect_outliers(phenotype["phenotype"], stratify=frame["Sex"])
        phenotype.loc[outliers, "phenotype"] = np.nan
        write_phenotype(phenotype, output_dir, name)
        if name in ADJUSTED:
            write_phenotype(adjust_for_medication(phenotype, medication, name), output_dir, f"{name}_adj")

    # FEV1 and FVC must use the same units; the saved ratio is not log-transformed.
    ratio = frame[["FID", "IID"]].copy()
    ratio["phenotype"] = prepare_fev1_fvc_ratio(frame["20150-0.0"], frame["20151-0.0"], frame["Sex"])
    write_phenotype(ratio, output_dir, "FEV1_FVC")

    # Screen waist and hip separately, then screen their raw ratio.
    waist = frame["48-0.0"].mask(detect_outliers(frame["48-0.0"], stratify=frame["Sex"]))
    hip = frame["49-0.0"].mask(detect_outliers(frame["49-0.0"], stratify=frame["Sex"]))
    ratio["phenotype"] = waist / hip
    ratio.loc[detect_outliers(ratio["phenotype"], stratify=frame["Sex"]), "phenotype"] = np.nan
    write_phenotype(ratio, output_dir, "WHR")
    print(f"Wrote {len(names)} UKB phenotype files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular-file", type=Path, required=True)
    parser.add_argument("--withdrawals-file", type=Path, required=True)
    parser.add_argument("--medication-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_phenotypes(args.tabular_file, args.withdrawals_file, args.medication_file, args.output_dir)


if __name__ == "__main__":
    main()
