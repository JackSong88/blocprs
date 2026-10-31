"""Assign both cohorts to windows using UKB reference-build positions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import normalize_chr


# Fixed windows are 1-based and inclusive; reject overlapping definitions.
def load_windows(path, chromosome):
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame["CHR"] = frame["CHR"].map(normalize_chr)
    frame = frame.loc[frame["CHR"] == chromosome].copy()
    for column in ("window_id_chr", "window_id_global", "start_bp", "end_bp"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame = frame.sort_values("start_bp").reset_index(drop=True)

    if frame.empty or frame["window_id_global"].duplicated().any():
        raise ValueError("Windows must be nonempty and have unique global IDs.")

    starts, ends = frame["start_bp"].to_numpy(), frame["end_bp"].to_numpy()

    if np.any(starts < 1) or np.any(ends < starts) or np.any(starts[1:] <= ends[:-1]):
        raise ValueError("Window boundaries must be positive, ordered and nonoverlapping.")

    return frame


# Transfer the reference assignment, not a fresh binning of GRCh38 positions.
def assign_windows(crosswalk, windows):
    positions = pd.to_numeric(crosswalk["ukbb_position"], errors="raise").to_numpy(dtype=np.int64)
    starts, ends = windows["start_bp"].to_numpy(), windows["end_bp"].to_numpy()

    # Find the last start <= position, then check the inclusive end below.
    indices = np.searchsorted(starts, positions, side="right") - 1

    if not ((indices >= 0) & (positions <= ends[np.maximum(indices, 0)])).all():
        raise ValueError("Some shared variants fall outside the reference windows.")

    result = crosswalk.copy()
    for column in ("window_id_chr", "window_id_global"):
        result[column] = windows[column].to_numpy()[indices]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--shared-crosswalk", type=Path, required=True)
    parser.add_argument("--windows", type=Path, default=Path("tables/windows_GRCh37_1Mb.tsv"),
                        help="Reference intervals; defaults to the paper's GRCh37 1-Mb windows.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chromosome = normalize_chr(args.chromosome)
    crosswalk = pd.read_csv(args.shared_crosswalk, sep="\t", dtype=str, keep_default_na=False)
    crosswalk = crosswalk.loc[crosswalk["ukbb_chr"].map(normalize_chr) == chromosome]

    if crosswalk.empty:
        raise ValueError("No shared variants for the requested chromosome.")

    assigned = assign_windows(crosswalk, load_windows(args.windows, chromosome))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(args.out, sep="\t", index=False)
    print(f"chr{chromosome}: assigned {len(assigned):,} source variants -> {args.out}")


if __name__ == "__main__":
    main()
