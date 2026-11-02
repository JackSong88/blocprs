"""Collects PLINK score sums into a participants x windows x PRSs array
used for downstream data creation.

Reads scores in chunks, saves completed window blocks for interrupted runs,
and writes scores.npy with participant and PRS labels for dataset creation.
"""

import argparse
import errno
import fcntl
import gzip
import json
import re
import shutil
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonical_transfer_utils import normalize_accession, parse_chromosomes

# Saves up to 32 windows per block so completed extraction work can be reused after interruption.
WINDOWS_PER_BLOCK = 32

# Caps the input and converted-output buffers together at 64 MiB (64 x 1024 x 1024 bytes).
OUTPUT_BUFFER_BYTES = 64 * 1024**2


# Describes a scoring result and the PRSs expected from its original weight file.
@dataclass
class ScoreFile:
    path: Path
    window: int | None
    sources: list[str]
    cohort: str
    chromosome: int


# Finds the PLINK scores for one additive_0, additive_1, etc. weight file.
# Returns the PRS names and window/chromosome labels used to group those scores.
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
    return ScoreFile(present[0], window, sources, match[1], int(match[2]))


# Finds the SUM column for each requested PRS.
def score_columns(path, requested, expected):
    header = pd.read_csv(path, sep=r"\s+", nrows=0).columns
    fid_column = "#FID" if "#FID" in header else "FID"
    selected = {}

    for column in header:

        if not column.endswith("_SUM"):
            continue

        source = normalize_accession(column[:-4])

        if source not in requested:
            continue

        if source in selected.values():
            raise ValueError(f"Duplicate SUM columns for {source} in {path}.")

        selected[column] = source

    if (set(expected) & set(requested)) != set(selected.values()):
        raise ValueError(f"SUM columns do not match the supplied scorefile: {path}")

    return fid_column, selected


# Reads participant rows in chunks and keeps only IDs and the requested PRS sums.
def score_chunks(path, requested, expected, block_size):
    fid_column, selected = score_columns(path, requested, expected)
    present = list(selected.values())
    dtype = {fid_column: str, "IID": str, **{column: np.float64 for column in selected}}
    reader = pd.read_csv(path, sep=r"\s+", usecols=[fid_column, "IID", *selected],
                         dtype=dtype, chunksize=block_size)

    with reader:
        for frame in reader:
            frame.rename(columns={fid_column: "FID", **selected}, inplace=True)
            yield frame, present


# Reads IDs from the first scoring result to establish the participant order
# that every subsequent scoring result must match.
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


# Checks all participant IDs for missing values and duplicates across chunks.
def validate_participants(ids):
    if ids.empty or ids.isna().any().any() or pd.MultiIndex.from_frame(ids).has_duplicates:
        raise ValueError("Missing or duplicate participant IDs.")


# Records input file details to check before reusing saved extraction blocks.
def file_signature(path):
    path = path.resolve()
    stat = path.stat()

    return [str(path), stat.st_size, stat.st_mtime_ns]


# Moves a completed file into place; copies it when working and output folders are on different disks.
def save_completed_file(source, destination):
    partial = destination.with_name(destination.name + ".partial")

    try:
        source.replace(partial)
    except OSError as error:

        if error.errno != errno.EXDEV:
            raise

        shutil.copyfile(source, partial)

    partial.replace(destination)


# Sums additive_0, additive_1, etc. scores for each participant, window and PRS.
# Saves the completed block so an interrupted extraction can reuse it.
def build_block(records, sources, windows, ids_frame, block_size, work_dir, destination):
    source_index = {source: index for index, source in enumerate(sources)}
    window_index = {window: index for index, window in enumerate(windows)}
    reference_fid = ids_frame["FID"].to_numpy()
    reference_iid = ids_frame["IID"].to_numpy()

    # Removes an unfinished working file left by an interrupted run.
    path = work_dir / "blocprs_extract.working.npy"
    path.unlink(missing_ok=True)

    try:
        # Stores all participants together for each window/PRS. This matches the
        # order read from PLINK and avoids scattered writes across a large array.
        scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                          shape=(len(ids_frame), len(windows), len(sources)), fortran_order=True)
        try:
            scores[:] = 0
            last_report = monotonic()

            for number, record in enumerate(records, start=1):
                start = 0
                window_position = window_index[record.window]

                for frame, present in score_chunks(record.path, sources, record.sources, block_size):
                    stop = start + len(frame)

                    if (not np.array_equal(frame["FID"].to_numpy(), reference_fid[start:stop])
                        or not np.array_equal(frame["IID"].to_numpy(), reference_iid[start:stop])):
                        raise ValueError(f"Participant order/membership differs in {record.path}; align before extraction.")

                    for source in present:
                        values = frame[source].to_numpy(dtype=np.float64)

                        if not np.isfinite(values).all():
                            raise ValueError(f"Missing or nonfinite scores in {record.path}")

                        # Sums this PRS's contributions from additive_0, additive_1, etc.
                        prs_position = source_index[source]
                        scores[start:stop, window_position, prs_position] += values

                    start = stop

                    now = monotonic()
                    if now - last_report >= 30:
                        print(f"Summing file {number}/{len(records)} in this block: "
                              f"{stop:,}/{len(ids_frame):,} participant rows.", flush=True)
                        last_report = now

                if start != len(ids_frame):
                    raise ValueError(f"Participant count differs in {record.path}.")

            scores.flush()
        finally:
            del scores

        save_completed_file(path, destination)
    finally:
        path.unlink(missing_ok=True)


# Checks whether a saved extraction block has the expected format and can be reused.
def complete_block(path, shape):
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = values.shape == shape and values.dtype == np.float64 and values.flags.f_contiguous
        del values
        return valid
    except (OSError, ValueError, EOFError):
        return False


# Joins completed window blocks without loading the full array into memory.
def write_output(block_paths, groups, sources, windows, ids_frame, output, block_size, *, dtype=np.float32):

    dtype = np.dtype(dtype)
    n_participants = len(ids_frame)
    values_per_chunk = max(1, min(block_size * WINDOWS_PER_BLOCK,
                                  OUTPUT_BUFFER_BYTES // (np.dtype(np.float64).itemsize + dtype.itemsize)))

    features = [("genomewide" if window is None else f"w{window:04d}", source,
                 source if window is None else f"{source}_w{window:04d}")
                for window in windows for source in sources]

    # Writes the score array and labels together before moving to the final directory.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", suffix=".partial", dir=output.parent) as directory:
        temporary_output = Path(directory)
        scores_path = temporary_output / "scores.npy"
        scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=dtype,
                                          shape=(n_participants, len(windows), len(sources)), fortran_order=True)
        output_offset = scores.offset
        scores.flush()
        del scores

        window_start = 0
        started = last_report = monotonic()
        written = 0
        total_values = n_participants * len(windows) * len(sources)

        for path, group in zip(block_paths, groups):
            block = np.load(path, mmap_mode="r", allow_pickle=False)
            input_offset = block.offset
            del block

            # Reads each PRS's adjacent participant-by-window values in order.
            # Seeks to that PRS's first window in the final array.
            with path.open("rb") as input_file, scores_path.open("r+b") as output_file:
                input_file.seek(input_offset)
                values_per_source = n_participants * len(group)

                for source_index in range(len(sources)):
                    output_start = (source_index * len(windows) + window_start) * n_participants
                    output_file.seek(output_offset + output_start * dtype.itemsize)

                    for start in range(0, values_per_source, values_per_chunk):
                        count = min(values_per_chunk, values_per_source - start)
                        values = np.fromfile(input_file, dtype=np.float64, count=count)

                        if len(values) != count:
                            raise ValueError(f"Incomplete saved score block: {path}")

                        # Converts only after adding all partitions and rejects overflow
                        # from either summation or float32 conversion.
                        with np.errstate(over="ignore", invalid="ignore"):
                            converted = values.astype(dtype)

                        if not np.isfinite(converted).all():
                            raise ValueError("Nonfinite scores after conversion; final output was not saved.")

                        converted.tofile(output_file)
                        del values, converted

                        written += count
                        now = monotonic()
                        if now - last_report >= 30 or written == total_values:
                            print(f"Writing scores.npy: {100 * written / total_values:.1f}%; "
                                  f"elapsed={now - started:.1f}s.", flush=True)
                            last_report = now

            window_start += len(group)

        # Saves labels in the same participant, window and source order.
        ids_frame[["FID", "IID"]].to_csv(temporary_output / "participants.tsv", sep="\t", index=False)
        pd.DataFrame(features, columns=["window", "source", "column"]).to_csv(
            temporary_output / "features.tsv", sep="\t", index=False)

        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output already exists: {output}")

        temporary_output.rename(output)


# Prepares saved input details, participant IDs and block paths for a new or resumed run.
# Locks the temporary directories and removes checkpoints only after successful output.
@contextmanager
def extraction_checkpoint(tmp_dir, work_dir, state, n_blocks, reference_file, block_size):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    block_paths = [tmp_dir / f"block_{index:05d}.npy" for index in range(n_blocks)]
    state_path, ids_path = tmp_dir / "state.json", tmp_dir / "ids.tsv"

    # Protects saved blocks and the working file when another run uses either directory.
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

        # Reuses blocks only when the input files, PRSs and window order still match.
        if state_path.exists():

            if json.loads(state_path.read_text()) != state:
                raise ValueError(f"Checkpoint inputs or settings differ in {tmp_dir}; use a new --tmp-dir.")

        else:

            if ids_path.exists() or any(tmp_dir.glob("block_*.npy")):
                raise ValueError(f"Checkpoints without compatible metadata in {tmp_dir}; use a new --tmp-dir.")

            partial = state_path.with_suffix(".json.partial")
            partial.write_text(json.dumps(state))
            partial.replace(state_path)

        # Restores the participant order used by saved blocks, or reads it from the first score file.
        if ids_path.exists():
            ids_frame = pd.read_csv(ids_path, sep="\t", dtype=str)
            validate_participants(ids_frame)
        else:

            if any(path.exists() for path in block_paths):
                raise ValueError("Checkpoint participant IDs are missing; use a new --tmp-dir.")

            ids_frame = reference_participants(reference_file, block_size)
            partial = ids_path.with_suffix(".tsv.partial")
            ids_frame.to_csv(partial, sep="\t", index=False)
            partial.replace(ids_path)

        # Keeps both locks held while extract_scores builds blocks and writes the output.
        yield ids_frame, block_paths

        # Runs only after successful output; leaves unrelated temporary files untouched.
        for path in [*block_paths, ids_path, state_path]:
            path.unlink()
            path.with_name(path.name + ".partial").unlink(missing_ok=True)

        (work_dir / "blocprs_extract.working.npy").unlink(missing_ok=True)


# Collects PLINK scores into one participant x window x PRS dataset.
# Saves completed blocks so an interrupted extraction can resume.
def extract_scores(scorefiles, score_dir, sources, output, block_size=65536, chromosomes=None,
                   tmp_dir=Path("data/tmp/extract_scores"), work_dir=None):

    if len(set(sources)) != len(sources):
        raise ValueError("Provide unique source names.")

    assert block_size >= 1

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    # Matches each weight file to its PLINK output and reads the PRS names.
    print(f"Reading score headers and locating {len(scorefiles):,} PLINK outputs.", flush=True)
    results = [scorefile_info(path, score_dir) for path in scorefiles]

    if chromosomes is not None and {record.chromosome for record in results} != set(chromosomes):
        raise ValueError("Supplied scorefiles do not cover exactly the requested chromosomes.")

    if len({record.path for record in results}) != len(results) or len({record.cohort for record in results}) != 1:
        raise ValueError("Scoring inputs must refer to distinct outputs from one cohort.")

    # Windowed inputs stay separate; full-chromosome inputs sum into one genome-wide score per PRS.
    full = all(record.window is None for record in results)

    if not full and any(record.window is None for record in results):
        raise ValueError("Do not combine windowed and whole-chromosome scoring inputs.")

    windows = [None] if full else sorted({record.window for record in results})

    #each ID should belong to only one chr
    if not full:
        for window in windows:
            if len({record.chromosome for record in results if record.window == window}) != 1:
                raise ValueError("Window IDs must be genome-wide unique.")

    requested = set(sources)
    available = set().union(*(record.sources for record in results))
    missing = requested - available

    if missing:
        raise ValueError(f"Requested sources absent from all inputs: {missing}")

    # Reads only files containing a requested PRS, but retains the full window list.
    # A window without variants for these PRSs stays in the output with zero scores.
    relevant = [record for record in results if set(record.sources) & requested]
    groups = [windows[start:start + WINDOWS_PER_BLOCK] for start in range(0, len(windows), WINDOWS_PER_BLOCK)] # divide window list into blocks

    tmp_dir = Path(tmp_dir)
    work_dir = tmp_dir if work_dir is None else Path(work_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Records input details and output ordering so saved blocks cannot be reused for a different extraction.
    state = {"version": 2, "output": str(output.resolve()), "sources": list(sources),
             "windows": windows, "windows_per_block": WINDOWS_PER_BLOCK,
             "inputs": [[file_signature(path), file_signature(record.path)] for path, record in zip(scorefiles, results)]}

    # Restores the participant order and saved-block paths, or prepares them for a new run.
    with extraction_checkpoint(tmp_dir, work_dir, state, len(groups), relevant[0].path, block_size) as (ids_frame, block_paths):
        print(f"Extracting {len(ids_frame):,} participants, {len(sources)} sources, {len(groups)} blocks.", flush=True)

        # Processes all participants and requested PRSs for one group of windows at a time.
        for index, (group, destination) in enumerate(zip(groups, block_paths)):
            shape = (len(ids_frame), len(group), len(sources))

            # A completed block stores every participant-window-PRS score for this group.
            # Reuses that saved array instead of repeating the extraction and summation.
            reused = complete_block(destination, shape)

            if not reused:

                # Selects the scoring files for this group's windows.
                records = [record for record in relevant if record.window in group]

                # Sums additive_0, additive_1, etc. for every participant-window-PRS combination
                # in this group, checks participant order, and saves the results as a reusable block.
                build_block(records, sources, group, ids_frame, block_size, work_dir, destination)

            print(f"Block {index + 1}/{len(groups)}: {'reused' if reused else 'saved'}.", flush=True)

        # Checks again because upstream score files can change while extraction runs.
        if state["inputs"] != [[file_signature(path), file_signature(record.path)] for path, record in zip(scorefiles, results)]:
            raise ValueError("Scoring inputs changed during extraction; final output was not saved.")

        # Assembles the saved blocks into scores.npy, keeping windows and PRSs separate.
        # Also saves the participant IDs and window/PRS labels in the matching order.
        print("Writing score array and participant/PRS labels.", flush=True)
        write_output(block_paths, groups, sources, windows, ids_frame, output, block_size)

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
