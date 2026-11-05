""" Optional Sanity check.
Compare raw window sums with independently scored canonical genome-wide PRSs.

Inputs are binary dataset directories. This optional check reads saved arrays,
not upstream alleles or PLINK records. Default tolerances accommodate printed
scores and float32 storage; they are screening tolerances, not formal bounds
derived from the original scorefiles.
"""

import argparse
from time import monotonic
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_utils import ScoreTable


# Compares aligned raw scores before any training-derived standardization.
def check_window_sums(windowed, genomewide, atol=1e-4, rtol=1e-6):

    assert np.isfinite([atol, rtol]).all() and min(atol, rtol) >= 0, "Tolerances must be finite and nonnegative."

    windows, reference = ScoreTable(windowed), ScoreTable(genomewide)
    features, reference_features = windows.features, reference.features
    sources, reference_sources = windows.sources, reference.sources

    if (features["window"].eq("genomewide").any()
            or not reference_features["window"].eq("genomewide").all()
            or set(sources) != set(reference_sources)):
        raise ValueError("Expected windowed and genome-wide datasets with identical sources.")

    # Matches by participant ID rather than assuming identical row order.
    ids = pd.MultiIndex.from_frame(windows.ids)
    reference_ids = pd.MultiIndex.from_frame(reference.ids)
    positions = reference_ids.get_indexer(ids)

    if not len(ids) or len(ids) != len(reference_ids) or (positions < 0).any():
        raise ValueError("Participant sets must match exactly; no rows are dropped or zero-filled.")

    # Matches source order before comparing each expert's scores.
    source_positions = [reference_sources.index(source) for source in sources]
    maximum, squared_error = np.zeros(len(sources)), np.zeros(len(sources))
    failures = np.zeros(len(sources), dtype=np.int64)
    block_size = max(1, min(8192, 32 * 1024**2 // (8 * np.prod(windows.array.shape[1:]))))

    # Reads bounded blocks and accumulates in float64 to limit summation error.
    started = last_report = monotonic()
    for start in range(0, len(ids), block_size):
        stop = min(start + block_size, len(ids))
        values = np.asarray(windows.array[start:stop], dtype=np.float64)
        expected = np.asarray(reference.array[positions[start:stop], 0, :], dtype=np.float64)
        expected = expected[:, source_positions]

        if not np.isfinite(values).all() or not np.isfinite(expected).all():
            raise ValueError("Scores must be finite.")

        difference = values.sum(axis=1, dtype=np.float64) - expected
        error = np.abs(difference)
        maximum = np.maximum(maximum, error.max(axis=0))
        squared_error += np.square(difference).sum(axis=0)
        failures += (error > atol + rtol * np.abs(expected)).sum(axis=0)

        now = monotonic()
        if now - last_report >= 30 or stop == len(ids):
            print(f"Compared {stop:,}/{len(ids):,} participants; elapsed={now - started:.1f}s.", flush=True)
            last_report = now

    # Reports reconstruction error separately for each source.
    print(f"N={len(ids):,}; windows={windows.array.shape[1]}; atol={atol:g}; rtol={rtol:g}", flush=True)
    for source, largest, squared, failed in zip(sources, maximum, squared_error, failures):
        print(f"{source}: max_abs={largest:.8g}, RMSE={np.sqrt(squared / len(ids)):.8g}, "
              f"outside_tolerance={failed}", flush=True)

    if failures.any():
        raise ValueError("Window sums differ from independent genome-wide scores beyond tolerance.")

    print("PASS: canonical window sums reproduce the independent genome-wide scores.", flush=True)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windowed", type=Path, required=True)
    parser.add_argument("--genomewide", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    print("Comparing window sums with independently scored genome-wide PRSs.", flush=True)
    check_window_sums(args.windowed, args.genomewide, args.atol, args.rtol)


if __name__ == "__main__":
    main()
