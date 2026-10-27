"""Extract CARTaGENE cholesterol and blood-pressure medication indicators."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import check_unique_ids, prepare_output_directory


# Only response 1 indicates medication use.
def extract_medication_data(metadata_file, output):
    columns = {"file111": "IID", "HIGH_BLOOD_CHOLESTEROL_CUR_TRE": "chol_med",
               "HIGH_BP_CURRENTLY_TREATED": "bp_med"}
    frame = pd.read_csv(metadata_file, usecols=list(columns), dtype={"file111": str}).rename(columns=columns)
    check_unique_ids(frame)
    frame["chol_med"] = frame["chol_med"].eq(1)
    frame["bp_med"] = frame["bp_med"].eq(1)
    output = Path(output)
    prepare_output_directory(output.parent)
    frame[["IID", "chol_med", "bp_med"]].to_csv(output, sep="\t", index=False)
    print(f"Wrote medication indicators for {len(frame):,} participants to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract_medication_data(args.metadata_file, args.output)


if __name__ == "__main__":
    main()
