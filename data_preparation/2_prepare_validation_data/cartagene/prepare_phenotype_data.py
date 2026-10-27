"""Prepare the CARTaGENE continuous and binary phenotypes used in the paper.

Clinical exclusions and sex-stratified outlier screening precede splitting.
Restricted cohort tables must be supplied.
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

METADATA_COLUMNS = {
    "file111": "IID", "sex_birth": "Sex", "DIABETES_T2": "T2D",
    "RES_MEASURED_FEV1": "FEV1", "RES_MEASURED_FVC": "FVC",
    "RES_BODY_MASS_INDEX": "BMI", "CALC_AVG_HEIGHT_CM": "HEIGHT",
    "ASTHMA_OCCURRENCE": "ASTHMA", "DIABETES_T1": "T1D",
    "HEART_ATTACK_OCCURRENCE": "CAD1", "ANGINA_OCCURRENCE": "CAD2",
    "ATHEROSCLEROSIS_OCCURRENCE": "CAD3", "HEART_DISEASE_OCCURRENCE": "CAD4",
    "HEART_FAILURE_OCCURRENCE": "HF", "ATRIAL_FIBRILLATION_OCCURRENCE": "AF",
    "STROKE_OCCURRENCE": "STR", "HIGH_BP_OCCURRENCE": "HTN", "GOUT_OCCURRENCE": "GOUT",
    "CALC_AVG_DIASTOLIC_BP": "DBP", "CALC_AVG_SYSTOLIC_BP": "SBP", "CALC_WAIST_TO_HIP_RATIO": "WHR",
}
CLINICAL_RANGES = {"HEIGHT": (130, 220), "BMI": (15, 60), "DBP": (40, 150),
                   "SBP": (70, 250), "WHR": (0.5, 1.5)}
BINARY_TRAITS = ["ASTHMA", "STR", "T2D", "T1D", "HTN", "GOUT", "AF", "HF"]
BINARY_RECODE = {2.0: 0.0, 9.0: np.nan, 99.0: np.nan, -7.0: np.nan, -9.0: np.nan}
BLOOD_TRAITS = {"LDL": "LDL", "HDL": "LOG_HDL", "TC": "TC", "TRIG": "LOG_TG",
                "CREATININE": "LOG_CRTN", "uric_acid": "URT"}
ADJUSTED = {"LDL", "TC", "SBP", "DBP"}


def continuous_phenotype(frame, field, name, limits=None):
    phenotype = frame[["FID", "IID", field]].rename(columns={field: "phenotype"}).copy()
    values = phenotype["phenotype"].replace({-9: np.nan, 999: np.nan})
    if limits is not None:
        values = values.mask((values < limits[0]) | (values > limits[1]))
    if field == "WHR":
        values = values.replace({9: np.nan})
    if name.startswith("LOG_"):
        values = np.log(values)
    phenotype["phenotype"] = values.mask(detect_outliers(values, stratify=frame["Sex"]))
    return phenotype


def prepare_phenotypes(metadata_file, genetic_codes_file, biochemistry_file, medication_file, output_dir):
    frame = pd.read_csv(metadata_file, usecols=list(METADATA_COLUMNS), dtype={"file111": str})
    frame = frame.rename(columns=METADATA_COLUMNS)
    check_unique_ids(frame)
    frame["FID"] = frame["IID"]
    medication = read_medication_use(medication_file)
    names = ["HEIGHT", "LOG_BMI", "DBP", "SBP", "WHR", "FEV1_FVC"] + BINARY_TRAITS + ["CAD"]
    names += list(BLOOD_TRAITS.values()) + [f"{name}_adj" for name in sorted(ADJUSTED)]
    output_dir = prepare_output_directory(output_dir)

    for field, limits in CLINICAL_RANGES.items():
        name = "LOG_BMI" if field == "BMI" else field
        phenotype = continuous_phenotype(frame, field, name, limits)
        write_phenotype(phenotype, output_dir, name)
        if name in ADJUSTED:
            write_phenotype(adjust_for_medication(phenotype, medication, name), output_dir, f"{name}_adj")

    # The shared lung-function rule returns a raw ratio after log-space QC.
    components = frame[["FEV1", "FVC"]].replace({-9: np.nan, 999: np.nan})
    ratio = frame[["FID", "IID"]].copy()
    ratio["phenotype"] = prepare_fev1_fvc_ratio(components["FEV1"], components["FVC"], frame["Sex"])
    write_phenotype(ratio, output_dir, "FEV1_FVC")

    for name in BINARY_TRAITS:
        phenotype = frame[["FID", "IID", name]].rename(columns={name: "phenotype"}).copy()
        phenotype["phenotype"] = phenotype["phenotype"].replace(BINARY_RECODE)
        write_phenotype(phenotype, output_dir, name)

    # Any recorded CAD component can define a case; all missing stays missing.
    components = frame[["CAD1", "CAD2", "CAD3", "CAD4"]].replace(BINARY_RECODE)
    cad = frame[["FID", "IID"]].copy()
    cad["phenotype"] = components.sum(axis=1, min_count=1).clip(lower=0, upper=1)
    write_phenotype(cad, output_dir, "CAD")

    codes = pd.read_csv(genetic_codes_file, dtype={"PROJECT_CODE": str, "file_111": str})
    blood = pd.read_csv(biochemistry_file, dtype={"PROJECT_CODE": str})
    blood = codes.merge(blood, on="PROJECT_CODE").rename(columns={"file_111": "IID"})
    blood["FID"] = blood["IID"]
    blood = blood.merge(frame[["IID", "Sex"]], on="IID", validate="many_to_one")
    check_unique_ids(blood)

    for field, name in BLOOD_TRAITS.items():
        phenotype = continuous_phenotype(blood, field, name)
        write_phenotype(phenotype, output_dir, name)
        if name in ADJUSTED:
            write_phenotype(adjust_for_medication(phenotype, medication, name), output_dir, f"{name}_adj")
    print(f"Wrote {len(names)} CARTaGENE phenotype files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--genetic-codes-file", type=Path, required=True)
    parser.add_argument("--biochemistry-file", type=Path, required=True)
    parser.add_argument("--medication-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_phenotypes(args.metadata_file, args.genetic_codes_file, args.biochemistry_file,
                       args.medication_file, args.output_dir)


if __name__ == "__main__":
    main()
