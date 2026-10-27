"""Extract UKB cholesterol and blood-pressure medication indicators."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import check_unique_ids, prepare_output_directory


def extract_medication_data(tabular_file, output):
    columns = pd.read_csv(tabular_file, nrows=0).columns
    medication_columns = [c for c in columns if "6177-0" in c or "6153-0" in c]

    if not medication_columns:
        raise ValueError("No baseline UKB medication fields 6177 or 6153 were found.")

    frame = pd.read_csv(tabular_file, usecols=["eid", "22001-0.0", "21022-0.0"] + medication_columns,
                        dtype={"eid": str})
    frame = frame.rename(columns={"eid": "IID", "22001-0.0": "Sex", "21022-0.0": "Age"})
    check_unique_ids(frame)
    frame = frame.dropna(subset=["Age", "Sex"])

    # Missing or unselected medication responses are not coded as medication use.
    frame["chol_med"] = frame[medication_columns].eq(1).any(axis=1)
    frame["bp_med"] = frame[medication_columns].eq(2).any(axis=1)
    output = Path(output)
    prepare_output_directory(output.parent)
    frame[["IID", "chol_med", "bp_med"]].to_csv(output, sep="\t", index=False)
    print(f"Wrote medication indicators for {len(frame):,} participants to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract_medication_data(args.tabular_file, args.output)


if __name__ == "__main__":
    main()
