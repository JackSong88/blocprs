"""Combine PLINK SUM scores from canonical scorefiles into a raw binary score bundle."""

import argparse
import errno
import fcntl
import gzip
import json
import re
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import normalize_accession, parse_chromosomes


WINDOWS_PER_BLOCK = 32
OUTPUT_BUFFER_BYTES = 64 * 1024**2


# Derive expected result files from the scoring inputs, not from whichever outputs exist.
def scorefile_info(path, score_dir):
    match = re.fullmatch(r"(ukbb|cartagene)_([0-9]+)_additive_([0-9]+)_(w([0-9]+)|fullchr)\.scorefile\.gz", path.name)

    if not match:
        raise ValueError(f"Unexpected scorefile name: {path.name}")

    stem = path.name.removesuffix(".scorefile.gz")
    candidates = [score_dir / f"{stem}.sscore{suffix}" for suffix in ("", ".gz", ".zst")]
    present = [candidate for candidate in candidates if candidate.is_file()]

    if len(present) != 1:
        raise ValueError(f"Expected exactly one scoring output for {path}, found {present}.")

    with gzip.open(path, "rt") as handle:
        columns = handle.readline().rstrip("\r\n").split("\t")[2:]
    sources = [normalize_accession(column) for column in columns]

    if not sources or len(set(sources)) != len(sources):
        raise ValueError(f"Invalid source columns in {path}.")

    window = None if match[5] is None else int(match[5])
    return present[0], window, sources, match[1], int(match[2])


# Match requested sources to their raw SUM columns.
def score_columns(path, requested, expected):
    header = pd.read_csv(path, sep=r"\s+", nrows=0).columns
    fid_column = "#FID" if "#FID" in header else "FID"
    selected = {}
    for column in header:
        if column.endswith("_SUM"):
            source = normalize_accession(column[:-4])
            if source in requested:

                if source in selected.values():
                    raise ValueError(f"Duplicate SUM columns for {source} in {path}.")

                selected[column] = source
    needed = set(expected) & set(requested)

    if needed != set(selected.values()):
        raise ValueError(f"SUM columns do not match the supplied scorefile: {path}")

    return fid_column, selected


# Read bounded participant chunks; other PLINK output columns are not retained.
def score_chunks(path, requested, expected, block_size):
    fid_column, selected = score_columns(path, requested, expected)
    dtype = {fid_column: str, "IID": str, **{column: np.float64 for column in selected}}
    reader = pd.read_csv(path, sep=r"\s+", usecols=[fid_column, "IID", *selected],
                         dtype=dtype, chunksize=block_size)

    with reader:
        for frame in reader:
            frame = frame.rename(columns={fid_column: "FID", **selected})

            if frame[["FID", "IID"]].isna().any().any():
                raise ValueError(f"Missing participant IDs in {path}.")

            if not np.isfinite(frame[list(selected.values())].to_numpy(dtype=float)).all():
                raise ValueError(f"Missing or nonfinite scores in {path}; no participant zero-filling is allowed.")

            yield frame, list(selected.values())


# Retain the full-table interface used by the independent window-sum checker.
def load_scores(path, requested, expected):
    frames = []
    present = []
    for frame, present in score_chunks(path, requested, expected, 65536):
        frames.append(frame)

    if not frames:
        raise ValueError(f"No participants in {path}.")

    frame = pd.concat(frames, ignore_index=True)
    ids = pd.MultiIndex.from_frame(frame[["FID", "IID"]])

    if frame.empty or ids.has_duplicates or frame[["FID", "IID"]].isna().any().any():
        raise ValueError(f"Missing or duplicate participants in {path}.")

    return frame, ids, present


# Read and validate the shared participant order once.
def reference_participants(path, block_size):
    header = pd.read_csv(path, sep=r"\s+", nrows=0).columns
    fid_column = "#FID" if "#FID" in header else "FID"
    with pd.read_csv(path, sep=r"\s+", usecols=[fid_column, "IID"], dtype=str,
                     chunksize=block_size) as reader:
        frames = [frame.rename(columns={fid_column: "FID"})[["FID", "IID"]] for frame in reader]

    if not frames:
        raise ValueError(f"No participants in {path}.")

    ids = pd.concat(frames, ignore_index=True)
    validate_participants(ids)
    return ids


# Duplicate IDs are checked globally, including across reading chunks.
def validate_participants(ids):

    if ids.empty or ids.isna().any().any() or pd.MultiIndex.from_frame(ids).has_duplicates:
        raise ValueError("Missing or duplicate participant IDs.")


# Record input identity for checkpoint reuse without hashing large scoring files.
def file_signature(path):
    path = path.resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


# Publish on the destination filesystem, even when temporary files are stored elsewhere.
def publish_file(source, destination):
    partial = destination.with_name(destination.name + ".partial")
    try:
        source.replace(partial)
    except OSError as error:

        if error.errno != errno.EXDEV:
            raise

        shutil.copyfile(source, partial)
    partial.replace(destination)


# Accumulate neighbouring window/source columns in a small column-major block.
def build_block(records, sources, windows, ids_frame, block_size, work_dir, destination):
    source_index = {source: index for index, source in enumerate(sources)}
    window_index = {window: index for index, window in enumerate(windows)}
    reference_ids = ids_frame.to_numpy()
    path = work_dir / "blocprs_extract.working.npy"
    path.unlink(missing_ok=True)
    required_bytes = len(ids_frame) * len(sources) * len(windows) * 8 + 4096

    if shutil.disk_usage(work_dir).free < required_bytes:
        raise ValueError(f"Not enough working space in {work_dir}; this block needs {required_bytes / 1024**3:.2f} GiB.")

    try:

        # Column-major storage makes each source/window update contiguous on disk.
        scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                          shape=(len(ids_frame), len(sources) * len(windows)), fortran_order=True)
        try:
            scores[:] = 0
            for score_path, window, expected, _, _ in records:
                start = 0
                for frame, present in score_chunks(score_path, sources, expected, block_size):
                    stop = start + len(frame)

                    if not np.array_equal(frame[["FID", "IID"]].to_numpy(), reference_ids[start:stop]):
                        raise ValueError(f"Participant order/membership differs in {score_path}; align before extraction.")

                    for source in present:
                        column = source_index[source] * len(windows) + window_index[window]

                        # Sum additive partitions in float64 rather than averaging them.
                        scores[start:stop, column] += frame[source].to_numpy(dtype=np.float64)
                    start = stop

                if start != len(reference_ids):
                    raise ValueError(f"Participant count differs in {score_path}.")

            scores.flush()
        finally:
            del scores
        publish_file(path, destination)
    finally:
        path.unlink(missing_ok=True)


# Only complete blocks with the expected layout can be reused.
def complete_block(path, shape):
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = values.shape == shape and values.dtype == np.float64 and values.flags.f_contiguous
        del values
        return valid
    except (OSError, ValueError, EOFError):
        return False


# Copy contiguous source/window slabs, releasing both mappings after each bounded chunk.
def write_output(block_paths, groups, sources, windows, ids_frame, output, block_size, *, dtype=np.float32):

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    dtype = np.dtype(dtype)
    n_participants = len(ids_frame)
    values_per_chunk = max(1, min(block_size * WINDOWS_PER_BLOCK,
                                  OUTPUT_BUFFER_BYTES // (np.dtype(np.float64).itemsize + dtype.itemsize)))
    features = [("genomewide" if window is None else f"w{window:04d}", source,
                 source if window is None else f"{source}_w{window:04d}")
                for window in windows for source in sources]

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", suffix=".partial", dir=output.parent) as directory:
        temporary_output = Path(directory)
        scores_path = temporary_output / "scores.npy"
        scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=dtype,
                                          shape=(n_participants, len(windows), len(sources)), fortran_order=True)
        output_offset = scores.offset
        scores.flush()
        del scores
        window_start = 0
        for path, group in zip(block_paths, groups):
            block = np.load(path, mmap_mode="r", allow_pickle=False)
            input_offset = block.offset
            del block
            width = len(group)
            for source_index in range(len(sources)):

                # Input: (participant, source * block_width + local_window), Fortran order.
                # Output: (participant, global_window, source), Fortran order.
                input_start = source_index * width * n_participants
                output_start = (source_index * len(windows) + window_start) * n_participants
                for start in range(0, n_participants * width, values_per_chunk):
                    count = min(values_per_chunk, n_participants * width - start)
                    values = np.memmap(path, mode="r", dtype=np.float64,
                                       offset=input_offset + (input_start + start) * 8, shape=(count,))
                    try:

                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite scores in an accumulated block; final output was not published.")

                        target = np.memmap(scores_path, mode="r+", dtype=dtype,
                                           offset=output_offset + (output_start + start) * dtype.itemsize, shape=(count,))
                        try:

                            # Convert precision only after the float64 partition sums are complete.
                            with np.errstate(over="ignore", invalid="ignore"):
                                target[:] = values

                            if not np.isfinite(target).all():
                                raise ValueError("Scores overflowed the output dtype; final output was not published.")

                            target.flush()
                        finally:
                            del target
                    finally:
                        del values
            window_start += width
        ids_frame[["FID", "IID"]].to_csv(temporary_output / "participants.tsv", sep="\t", index=False)
        pd.DataFrame(features, columns=["window", "source", "column"]).to_csv(
            temporary_output / "features.tsv", sep="\t", index=False)

        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output already exists: {output}")

        temporary_output.rename(output)


# Reuse completed blocks after interruption and remove them after successful output.
def extract_scores(scorefiles, score_dir, sources, output, block_size=65536, chromosomes=None,
                   tmp_dir=Path("data/tmp/extract_scores"), work_dir=None):

    if not sources or len(set(sources)) != len(sources) or len(set(scorefiles)) != len(scorefiles):
        raise ValueError("Provide unique source names and scorefiles.")

    if not scorefiles or block_size < 1:
        raise ValueError("Provide a nonempty scorefile list and positive block size.")

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    info = [scorefile_info(path, score_dir) for path in scorefiles]

    if chromosomes is not None and {row[4] for row in info} != set(chromosomes):
        raise ValueError("Supplied scorefiles do not cover exactly the requested chromosomes.")

    if len({path for path, *_ in info}) != len(info) or len({row[3] for row in info}) != 1:
        raise ValueError("Scoring inputs must refer to distinct outputs from one cohort.")

    full = all(window is None for _, window, *_ in info)

    if not full and any(window is None for _, window, *_ in info):
        raise ValueError("Do not combine windowed and whole-chromosome scoring inputs.")

    windows = [None] if full else sorted({window for _, window, *_ in info})
    if not full:
        for window in windows:

            if len({row[4] for row in info if row[1] == window}) != 1:
                raise ValueError("Window IDs must be genome-wide unique.")

    available = set().union(*(set(row[2]) for row in info))

    if set(sources) - available:
        raise ValueError(f"Requested sources absent from all inputs: {set(sources) - available}")

    # Keep the full window layout, but do not parse irrelevant participant scores.
    relevant = [row for row in info if set(row[2]) & set(sources)]
    groups = [windows[start:start + WINDOWS_PER_BLOCK] for start in range(0, len(windows), WINDOWS_PER_BLOCK)]
    tmp_dir = Path(tmp_dir)
    work_dir = tmp_dir if work_dir is None else Path(work_dir)

    if output.resolve().is_relative_to(tmp_dir.resolve()) or output.resolve().is_relative_to(work_dir.resolve()):
        raise ValueError("Keep the final output outside the temporary and working directories.")

    if output.resolve() in {path.resolve() for path in [*scorefiles, *(row[0] for row in info)]}:
        raise ValueError("The final output must not replace a scoring input.")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    block_paths = [tmp_dir / f"block_{index:05d}.npy" for index in range(len(groups))]
    state_path, ids_path = tmp_dir / "state.json", tmp_dir / "ids.tsv"
    state = {"version": 1, "output": str(output.resolve()), "sources": list(sources),
             "windows": windows, "windows_per_block": WINDOWS_PER_BLOCK,
             "inputs": [[file_signature(path), file_signature(row[0])] for path, row in zip(scorefiles, info)]}

    # Lock both directories; retain empty lock files so retries lock the same file.
    lock_paths = [tmp_dir / ".lock"]
    if work_dir.resolve() != tmp_dir.resolve():
        lock_paths.append(work_dir / ".lock")

    with ExitStack() as locks:
        for path in lock_paths:
            lock = locks.enter_context(path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Another extraction is using {path.parent}; choose a separate temporary directory.") from error

        if state_path.exists():

            if json.loads(state_path.read_text()) != state:
                raise ValueError(f"Checkpoint inputs or settings differ in {tmp_dir}; use a new --tmp-dir.")

        else:

            if ids_path.exists() or any(tmp_dir.glob("block_*.npy")):
                raise ValueError(f"Checkpoints without compatible metadata in {tmp_dir}; use a new --tmp-dir.")

            partial = state_path.with_suffix(".json.partial")
            partial.write_text(json.dumps(state))
            partial.replace(state_path)

        if ids_path.exists():
            ids_frame = pd.read_csv(ids_path, sep="\t", dtype=str)
            validate_participants(ids_frame)
        else:

            if any(path.exists() for path in block_paths):
                raise ValueError("Checkpoint participant IDs are missing; use a new --tmp-dir.")

            ids_frame = reference_participants(relevant[0][0], block_size)
            partial = ids_path.with_suffix(".tsv.partial")
            ids_frame.to_csv(partial, sep="\t", index=False)
            partial.replace(ids_path)

        print(f"Extracting {len(ids_frame):,} participants, {len(sources)} sources, {len(groups)} blocks.", flush=True)
        for index, (group, destination) in enumerate(zip(groups, block_paths)):
            shape = (len(ids_frame), len(sources) * len(group))
            reused = complete_block(destination, shape)

            if not reused:
                records = [row for row in relevant if row[1] in group]
                build_block(records, sources, group, ids_frame, block_size, work_dir, destination)

            print(f"Block {index + 1}/{len(groups)}: {'reused' if reused else 'saved'}.", flush=True)

        if state["inputs"] != [[file_signature(path), file_signature(row[0])] for path, row in zip(scorefiles, info)]:
            raise ValueError("Scoring inputs changed during extraction; final output was not replaced.")

        print("Writing binary score bundle.", flush=True)
        write_output(block_paths, groups, sources, windows, ids_frame, output, block_size)

        # Remove only this extractor's checkpoints, not unrelated files in --tmp-dir.
        for path in [*block_paths, ids_path, state_path]:
            path.unlink()
            path.with_name(path.name + ".partial").unlink(missing_ok=True)
        (work_dir / "blocprs_extract.working.npy").unlink(missing_ok=True)

    print(f"Wrote {len(ids_frame):,} participants x {len(sources) * len(windows):,} scores to {output}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorefiles", type=Path, nargs="+", required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory for scores.npy, participants.tsv, and features.tsv.")
    parser.add_argument("--block-size", type=int, default=65536, help="Participant rows per input chunk; binary copies are memory-capped.")
    parser.add_argument("--tmp-dir", type=Path, default=Path("data/tmp/extract_scores"),
                        help="Resumable block directory; use a separate directory for each output.")
    parser.add_argument("--work-dir", type=Path, help="Optional working directory, e.g. node-local storage; defaults to --tmp-dir.")
    parser.add_argument("--chromosomes", default="1-22", help="Expected chromosome coverage, e.g. 1-22 or 1.")
    args = parser.parse_args()
    chromosomes = [int(value) for value in parse_chromosomes(args.chromosomes)]
    extract_scores(args.scorefiles, args.score_dir, args.sources, args.out, args.block_size, chromosomes,
                   args.tmp_dir, args.work_dir)


if __name__ == "__main__":
    main()
