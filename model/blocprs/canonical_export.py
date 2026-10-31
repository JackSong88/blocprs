"""SNP-weight export from canonical variant-to-window mappings.

Final cross-mapping files supply source weights and window assignments.
Cohort PVAR metadata defines allele orientation; dosage reversals retain
their additive offsets. Export requires an explicit genome build and
autosomal diploid scoring.
"""

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .export import _model_arrays, _numeric_column, export_snp_weights


# Keep integer chromosome/position labels exact and restrict dosage reversal to autosomes.
def _coordinates(frame):
    chromosome = frame["chromosome"].astype(str).str.replace(r"^chr", "", regex=True)
    positions = _numeric_column(frame, "position")

    if not chromosome.isin([str(value) for value in range(1, 23)]).all():
        raise ValueError("Canonical SNP export requires autosomal diploid scoring (chromosomes 1-22).")

    if np.any((positions <= 0) | (positions != np.floor(positions)) | (positions >= 2**53)):
        raise ValueError("Canonical variant positions must be positive, exact integers below 2**53.")

    frame["chromosome"] = chromosome
    frame["position"] = positions.astype(np.int64)


# Read only fitted sources from the final, window-assigned crosswalk files.
def _read_cross_mapping(paths, sources, windows, cohort):
    columns = ["accession", "source_key", "effect_weight", "window_id_global",
               f"{cohort}_id", f"{cohort}_effect_allele", f"{cohort}_chr", f"{cohort}_position"]
    selected = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"))

        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate crosswalk column names in {path}.")

        with pd.read_csv(path, sep="\t", usecols=columns, dtype=str,
                         keep_default_na=False, chunksize=100000) as reader:
            for frame in reader:
                frame["accession"] = frame["accession"].str.replace(r"_hmPOS_GRCh(?:37|38)$", "", regex=True)
                frame = frame.loc[frame["accession"].isin(sources)].copy()
                if frame.empty:
                    continue

                frame = frame.rename(columns={"accession": "source", "effect_weight": "weight",
                                               f"{cohort}_id": "variant_id", f"{cohort}_effect_allele": "matched_allele",
                                               f"{cohort}_chr": "chromosome", f"{cohort}_position": "position"})
                ids = _numeric_column(frame, "window_id_global")

                if np.any((ids < 0) | (ids != np.floor(ids)) | (ids >= 2**53)):
                    raise ValueError("The crosswalk needs nonnegative, integer window_id_global assignments.")

                frame["window"] = [f"w{int(value):04d}" for value in ids]

                if not set(frame["window"]).issubset(windows):
                    raise ValueError("Canonical window assignments do not match the fitted model's window labels.")

                frame["weight"] = _numeric_column(frame, "weight")
                _coordinates(frame)
                selected.append(frame.drop(columns="window_id_global"))

    if not selected:
        raise ValueError("The supplied crosswalk contains no fitted sources.")

    mapping = pd.concat(selected, ignore_index=True)

    if mapping["source_key"].eq("").any() or mapping.duplicated(["source", "source_key"]).any():
        raise ValueError("Canonical source identities are missing or duplicated across crosswalk files.")

    if mapping.duplicated(["source", "variant_id"]).any():
        raise ValueError("A fitted source/variant appears more than once in the crosswalk.")

    if set(mapping["source"]) != set(sources):
        raise ValueError("The supplied crosswalk must cover every fitted source.")

    return mapping.drop(columns="source_key")


# Reconstruct the exact CHR:POS:REF:ALT IDs used by the canonical PLINK scoring script.
def _read_pvar(path, requested, max_allele_length):
    columns = ["chromosome", "position", "raw_id", "ref", "alt"]
    wanted_ids = set(requested["variant_id"])
    wanted_positions = set(requested["position"])
    selected = []

    # PVAR's first five fields are CHROM, POS, ID, REF and ALT; metadata starts with '#'.
    with pd.read_csv(path, sep="\t", comment="#", header=None, names=columns,
                     usecols=range(5), dtype=str, keep_default_na=False, chunksize=100000) as reader:
        for frame in reader:
            _coordinates(frame)
            frame = frame.loc[frame["position"].isin(wanted_positions)].copy()
            if frame.empty:
                continue

            if frame["alt"].str.contains(",", regex=False).any():
                raise ValueError(f"Multiallelic genotype metadata at a requested position in {path}; split/harmonize before export.")

            frame["variant_id"] = (frame["chromosome"] + ":" + frame["position"].astype(str)
                                   + ":" + frame["ref"] + ":" + frame["alt"])
            frame = frame.loc[frame["variant_id"].isin(wanted_ids)].copy()
            if not frame.empty:
                selected.append(frame.drop(columns=["raw_id", "chromosome", "position"]))

    if not selected:
        raise ValueError(f"No required canonical scorer IDs were found in {path}.")

    alleles = pd.concat(selected, ignore_index=True)

    if alleles["variant_id"].duplicated().any() or set(alleles["variant_id"]) != wanted_ids:
        raise ValueError(f"Missing or duplicated canonical scorer IDs in {path}.")

    if (alleles["ref"].isin(["", "."]) | alleles["alt"].isin(["", "."]) | alleles["ref"].eq(alleles["alt"])).any():
        raise ValueError(f"Missing or identical REF/ALT alleles in {path}.")

    if (alleles["ref"].str.len() > max_allele_length).any() or (alleles["alt"].str.len() > max_allele_length).any():
        raise ValueError(f"A required allele exceeds the canonical scorer's ID-length limit in {path}.")

    return alleles


# Normalize to cohort ALT dosage, retaining the diploid constant when REF was scored.
def export_canonical_snp_weights(model, cross_mapping, *, cohort, genome_build, pvar_template=None, path=None):
    """Use assigned crosswalk weights and cohort PVAR alleles; no new matching or liftover.

    Inputs must be the complete canonical panel used for the fitted scores.
    Metadata is read in chunks; selected source rows and final SNP weights remain in memory.
    The stated build is required because neither PVAR nor the crosswalk guarantees build metadata.
    """

    if cohort not in ("ukbb", "cartagene") or genome_build not in ("GRCh37", "GRCh38"):
        raise ValueError("Specify cohort ukbb/cartagene and the actual PVAR genome build GRCh37/GRCh38.")

    paths = [Path(cross_mapping)] if isinstance(cross_mapping, (str, Path)) else [Path(item) for item in cross_mapping]

    if not paths or len({item.resolve() for item in paths}) != len(paths):
        raise ValueError("Supply each final assigned crosswalk file exactly once.")

    sources, windows, *_ = _model_arrays(model)
    mapping = _read_cross_mapping(paths, sources, windows, cohort)
    template = str(pvar_template or f"data/{cohort}/genotypes/chr{{chr}}.pvar")

    if "{chr}" not in template:
        raise ValueError("pvar_template must contain {chr} for the cohort chromosome files.")

    oriented = []

    # Match the ID-length limits in score_canonical_transfer_windows.sh.
    max_allele_length = 100 if cohort == "ukbb" else 1000
    for chromosome, frame in mapping.groupby("chromosome", sort=False):
        alleles = _read_pvar(Path(template.format(chr=chromosome)), frame, max_allele_length)
        frame = frame.merge(alleles, on="variant_id", how="left", validate="many_to_one", sort=False)
        reversed_allele = frame["matched_allele"].eq(frame["ref"])

        if not (reversed_allele | frame["matched_allele"].eq(frame["alt"])).all():
            raise ValueError("A matched effect allele is neither REF nor ALT in the cohort PVAR.")

        # beta * dosage(REF) = 2 * beta - beta * dosage(ALT).
        # Keep the constant as well as flipping the weight, so exported scores retain their level.
        frame["allele_offset"] = np.where(reversed_allele, 2 * frame["weight"], 0.0)
        frame["weight"] = np.where(reversed_allele, -frame["weight"], frame["weight"])
        frame["effect_allele"], frame["other_allele"] = frame["alt"], frame["ref"]
        frame["genome_build"] = genome_build
        oriented.append(frame.drop(columns=["matched_allele", "ref", "alt"]))

    return export_snp_weights(model, pd.concat(oriented, ignore_index=True), path)
