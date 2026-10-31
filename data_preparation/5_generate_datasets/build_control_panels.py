"""Combine real target and out-of-phenotype PRSs on the same participants.

The first input defines participant order. Every selected source must cover
those IDs and the same windows; missing scores are never replaced with zero.
The output is a raw score directory for create_datasets.py, not a new split.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from create_datasets import feature_layout, participant_index
from score_io import ScoreTable


# Read all ID labels without loading the score matrix.
def score_ids(table):
    ids = pd.concat(list(table.iter_ids()), ignore_index=True)
    participant_index(ids, "Input scores")
    return ids


# Resolve selected sources and require identical window definitions.
def panel_layout(inputs, selections):
    plans, output_sources = [], []
    reference_windows = feature_layout(next(iter(inputs.values()))).window.drop_duplicates().tolist()
    for specification in selections:
        parts = specification.split(":")

        if len(parts) not in (2, 3):
            raise ValueError("Use INPUT:SOURCE or INPUT:SOURCE:OUTPUT_NAME.")

        name, source = parts[:2]
        alias = parts[2] if len(parts) == 3 else source
        features = feature_layout(inputs[name])
        selected = features.loc[features.source.eq(source)]

        if selected.empty:
            raise ValueError(f"Source {source!r} is absent from input {name!r}.")

        windows = selected.window.tolist()
        if windows != reference_windows:
            raise ValueError("Sources must have the same ordered windows or all be genome-wide.")

        if not alias.strip() or alias != alias.strip() or alias.endswith("_") or alias in output_sources:
            raise ValueError("Source names must be unique, without surrounding spaces or trailing underscores.")

        output_sources.append(alias)
        plans.append((name, selected.column.tolist()))

    rows = [(window, source, source if window == "genomewide" else f"{source}_{window}")
            for window in reference_windows for source in output_sources]
    return plans, pd.DataFrame(rows, columns=["window", "source", "column"])


# Stream selected score columns into one aligned float32 panel.
def build_panel(inputs, selections, output_dir):
    if not inputs or not selections:
        raise ValueError("Provide at least one input and one selected source.")

    output_dir = Path(output_dir)

    if output_dir.exists():
        raise FileExistsError(output_dir)

    plans, features = panel_layout(inputs, selections)
    tables = {name: ScoreTable(path) for name, path in inputs.items()}

    # Keep the first input's population even when none of its sources are selected.
    ids = score_ids(next(iter(tables.values())))
    reference = participant_index(ids, "Reference scores")
    n_windows, n_sources = features.window.nunique(), features.source.nunique()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".control_panel_", dir=output_dir.parent) as temporary:
        directory = Path(temporary) / "scores"
        directory.mkdir()
        scores = np.lib.format.open_memmap(directory / "scores.npy", mode="w+", dtype=np.float32,
                                         shape=(len(ids), n_windows, n_sources))
        try:
            for name, table in tables.items():
                selected = [(index, columns) for index, (input_name, columns) in enumerate(plans)
                            if input_name == name]
                if not selected:
                    continue

                available = participant_index(score_ids(table), name)

                if not reference.isin(available).all():
                    raise ValueError(f"Input {name!r} lacks reference participants; align scores before making controls.")

                columns = list(dict.fromkeys(column for _, names in selected for column in names))
                positions = {column: index for index, column in enumerate(columns)}
                written = np.zeros(len(ids), dtype=bool)
                for block_ids, values in table.iter_chunks(columns=columns, dtype=np.float32):

                    # Reorder by ID; extra input participants do not enter the reference panel.
                    rows = reference.get_indexer(participant_index(block_ids, name))
                    keep = rows >= 0
                    rows, values = rows[keep], values[keep]

                    if written[rows].any() or not np.isfinite(values).all():
                        raise ValueError("Repeated participants or nonfinite scores in a control input.")

                    for source_index, names in selected:
                        scores[rows, :, source_index] = values[:, [positions[column] for column in names]]
                    written[rows] = True

                if not written.all():
                    raise ValueError(f"Input {name!r} changed between ID inspection and scoring.")

                scores.flush()
        finally:
            del scores

        ids.to_csv(directory / "participants.tsv", sep="\t", index=False)
        features.to_csv(directory / "features.tsv", sep="\t", index=False)
        directory.rename(output_dir)
    return len(ids), n_windows, n_sources


# Define source panels explicitly; do not select controls using held-out performance.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="NAME=raw_score_path; repeat, reference population first.")
    parser.add_argument("--sources", nargs="+", required=True, help="INPUT:SOURCE[:OUTPUT_NAME] selections.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    for specification in args.input:
        name, separator, path = specification.partition("=")

        if not separator or not name or not path or name in inputs:
            parser.error("Inputs must have distinct NAME=PATH assignments.")

        inputs[name] = Path(path)
    n, windows, sources = build_panel(inputs, args.sources, args.output_dir)
    print(f"N={n}, windows={windows}, sources={sources}; saved={args.output_dir}")


if __name__ == "__main__":
    main()
