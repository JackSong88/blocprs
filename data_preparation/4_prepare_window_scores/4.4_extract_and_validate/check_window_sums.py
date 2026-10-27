"""Check extracted window sums using the precision of their original PLINK SUM outputs."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from extract_pgsc_scores import load_scores, scorefile_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from score_io import ScoreTable


# PLINK writes six significant digits; each printed value contributes half a unit.
def rounding_bound(values):
    values = np.asarray(values, dtype=float)
    unit = np.zeros_like(values)
    nonzero = values != 0
    unit[nonzero] = np.power(10.0, np.floor(np.log10(np.abs(values[nonzero]))) - 5)
    scaled = np.divide(values, unit, out=np.zeros_like(values), where=nonzero)

    if not np.isfinite(values).all() or np.any(np.abs(scaled - np.rint(scaled)) > 1e-7):
        raise ValueError("Raw scores do not have the expected six-significant-digit precision.")

    return unit / 2


# Compare nonzero weights as strings, preserving allele and source identity.
def weight_entries(path, sources):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    ids, alleles = frame.iloc[:, 0], frame.iloc[:, 1]
    entries = {}
    for column in frame.columns[2:]:
        source = re.sub(r"_hmPOS_GRCh(?:37|38)$", "", column)
        if source not in sources:
            continue
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite weights: {path}")

        for index in np.flatnonzero(values):
            key = (ids.iloc[index], alleles.iloc[index], source)

            if key in entries:
                raise ValueError(f"Duplicate variant/allele/source: {path}")

            entries[key] = frame[column].iloc[index]

    return entries, set(ids)


# Group each chromosome's additive partitions without confusing them with windows.
def score_groups(paths, score_dir, full):
    groups = {}
    for path in paths:
        info = scorefile_info(path, score_dir)
        score_path, window, _, cohort, chrom = info

        if (window is None) != full:
            raise ValueError("Window and full-chromosome scorefile lists are mixed.")

        partition = int(re.search(r"_additive_(\d+)_", path.name)[1])
        groups.setdefault((cohort, chrom, partition), []).append((path, info))

    if len(paths) != len(set(paths)) or not groups:
        raise ValueError("Provide nonempty, unique scorefile lists.")

    return groups


# Read every raw partition once and keep only participant-by-source accumulators.
def raw_reference(window_files, full_files, score_dir, sources, chromosomes):
    window_groups = score_groups(window_files, score_dir, False)
    full_groups = score_groups(full_files, score_dir, True)

    if window_groups.keys() != full_groups.keys():
        raise ValueError("Window/full-chromosome partition coverage differs.")

    keys = set(window_groups)
    if len({key[0] for key in keys}) != 1 or {key[1] for key in keys} != set(chromosomes):
        raise ValueError("Scorefiles do not cover exactly one cohort and the requested chromosomes.")

    reference_ids = None
    total = reference = allowance = magnitude = None
    source_positions = {source: index for index, source in enumerate(sources)}
    windows_seen = {}

    for key in sorted(keys):
        if len(full_groups[key]) != 1:
            raise ValueError(f"Multiple full-chromosome inputs for partition {key}.")

        full_path, _ = full_groups[key][0]
        expected_weights, full_variants = weight_entries(full_path, sources)
        seen_weights, seen_variants = set(), set()

        for is_full, records in ((True, full_groups[key]), (False, window_groups[key])):
            for path, (score_path, window, expected, _, chrom) in records:
                if not is_full:
                    previous = windows_seen.setdefault(window, chrom)

                    if previous != chrom:
                        raise ValueError("Window IDs must be genome-wide unique.")

                    weights, variants = weight_entries(path, sources)
                    if seen_weights & weights.keys() or any(expected_weights.get(k) != v for k, v in weights.items()):
                        raise ValueError(f"Repeated or changed window weights/alleles: {path}")

                    seen_weights.update(weights)
                    seen_variants.update(variants)
                else:
                    variants = full_variants

                variants_path = Path(str(score_path).split(".sscore")[0] + ".sscore.vars")
                scored_variants = set(variants_path.read_text().splitlines())

                if scored_variants != variants:
                    raise ValueError(f"Scored variants differ from the scorefile: {path}")

                frame, ids, present = load_scores(score_path, sources, expected)
                if reference_ids is None:
                    reference_ids = ids
                    shape = (len(ids), len(sources))
                    total, reference, allowance, magnitude = [np.zeros(shape) for _ in range(4)]

                if not reference_ids.equals(ids):
                    raise ValueError(f"Participant order/membership differs: {path}")

                positions = [source_positions[source] for source in present]
                values = frame[present].to_numpy(dtype=float)
                (reference if is_full else total)[:, positions] += values
                allowance[:, positions] += rounding_bound(values)
                magnitude[:, positions] += np.abs(values)

        if seen_weights != expected_weights.keys() or seen_variants != full_variants:
            raise ValueError(f"Incomplete window weights or variant coverage: {key}")

    # Also allow floating-point accumulation, separately from printed-score rounding.
    allowance += 64 * np.finfo(float).eps * (magnitude + 1)
    error = np.abs(total - reference)

    if np.any(error > allowance):
        raise ValueError("Raw window sums exceed the accumulated score-printing error bound.")

    print(f"Raw inputs agree: {len(reference_ids):,} participants, {len(sources)} sources, "
          f"{len(windows_seen)} windows.", flush=True)
    return reference_ids, total, reference, allowance, magnitude


# Allow only the rounding introduced by an explicitly float32 score array.
def storage_rounding(table, magnitude, n_terms=1):
    if table.array is None or table.array.dtype != np.float32:
        return 0.0
    unit = np.finfo(np.float32).eps / 2
    smallest = float(np.nextafter(np.float32(0), np.float32(1)))
    return unit * magnitude + n_terms * smallest


# Check extracted scores against raw PLINK totals, independently of dataset construction.
def check_window_sums(windowed, genomewide, ids, raw_total, raw_full, allowance, magnitude,
                      sources, block_size=8192):
    reference_table = ScoreTable(genomewide)
    frames = []
    for chunk_ids, values in reference_table.iter_chunks(chunk_size=block_size):
        frame = pd.DataFrame(values, columns=reference_table.columns)
        frame.index = pd.MultiIndex.from_frame(chunk_ids)
        frames.append(frame)
    reference = pd.concat(frames)

    if reference.index.has_duplicates or not reference.index.equals(ids):
        raise ValueError("Extracted reference participant order/membership differs from raw scoring.")

    selected_sources = reference.columns.tolist()
    positions = [sources.index(source) for source in selected_sources]
    expected = reference.to_numpy(dtype=float)
    numeric = 64 * np.finfo(float).eps * (magnitude[:, positions] + 1)
    reference_rounding = storage_rounding(reference_table, np.abs(raw_full[:, positions]))

    if not np.isfinite(expected).all() or np.any(np.abs(expected - raw_full[:, positions]) > numeric + reference_rounding):
        raise ValueError("Extracted reference differs from the original full-chromosome SUM scores.")

    window_table = ScoreTable(windowed)
    columns = window_table.columns
    groups = {source: [] for source in selected_sources}
    for column in columns:
        if column in ("FID", "IID"):
            continue
        match = re.fullmatch(r"(.+?)(?:__w|_w)([0-9]+)", column)

        if not match or match[1] not in groups:
            raise ValueError(f"Unexpected windowed source column: {column}")

        groups[match[1]].append(column)

    if not groups or any(not columns for columns in groups.values()):
        raise ValueError("Each source must have window columns.")

    offset = 0
    maximum = np.zeros(len(positions))
    indices = [[columns.index(column) for column in groups[source]] for source in selected_sources]
    window_rounding = storage_rounding(window_table, magnitude[:, positions], max(map(len, indices)))
    for chunk_ids, values in window_table.iter_chunks(chunk_size=block_size):
        stop = offset + len(chunk_ids)
        observed_ids = pd.MultiIndex.from_frame(chunk_ids)

        if not observed_ids.equals(ids[offset:stop]):
            raise ValueError("Extracted window participant order/membership differs from raw scoring.")

        if not np.isfinite(values).all():
            raise ValueError("Nonfinite extracted scores.")

        summed = np.column_stack([values[:, selected].sum(axis=1, dtype=np.float64) for selected in indices])
        quantized = window_rounding[offset:stop] if np.ndim(window_rounding) else window_rounding
        reference_quantized = reference_rounding[offset:stop] if np.ndim(reference_rounding) else reference_rounding
        if np.any(np.abs(summed - raw_total[offset:stop, positions]) > numeric[offset:stop] + quantized):
            raise ValueError("Extracted window sums differ from the original window SUM scores.")

        error = np.abs(summed - expected[offset:stop])
        if np.any(error > allowance[offset:stop, positions] + numeric[offset:stop] + quantized + reference_quantized):
            raise ValueError("Extracted window sums exceed the score-printing error bound.")

        maximum = np.maximum(maximum, error.max(axis=0))
        offset = stop

    if offset != len(ids):
        raise ValueError("Extracted windows omit participants.")

    for source, error in zip(selected_sources, maximum):
        print(f"{source}: N={offset:,}, max absolute difference={error:.8g}; rounding check passed.",
              flush=True)
    return dict(zip(selected_sources, maximum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windowed", type=Path, nargs="+", required=True)
    parser.add_argument("--genomewide", type=Path, nargs="+", required=True)
    parser.add_argument("--window-scorefiles", type=Path, nargs="+", required=True)
    parser.add_argument("--fullchr-scorefiles", type=Path, nargs="+", required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--chromosomes", default="1-22")
    args = parser.parse_args()

    if len(args.windowed) != len(args.genomewide):
        parser.error("Provide a matching reference table for each window table.")

    sources = sorted(set().union(*(set(ScoreTable(path).columns) for path in args.genomewide)))
    from canonical_transfer_utils import parse_chromosomes
    chromosomes = [int(value) for value in parse_chromosomes(args.chromosomes)]
    raw = raw_reference(args.window_scorefiles, args.fullchr_scorefiles, args.score_dir, sources, chromosomes)
    for windowed, genomewide in zip(args.windowed, args.genomewide):
        check_window_sums(windowed, genomewide, *raw, sources)


if __name__ == "__main__":
    main()
