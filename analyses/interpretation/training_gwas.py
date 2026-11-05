"""Run restricted-variant GWAS on the saved training participants.

Inputs are processed GRCh37 PGEN files and the stage-2 scoring-position union.
Stored hard calls match the restricted GWAS concentration analysis. Phenotypes
retain their dataset transformation without further scaling. Only participant
metadata and covariates are read; PRS arrays are not loaded.
"""

import argparse
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from participant_utils import filter_participants, id_positions, read_participants


# Participant selection starts from stored training rows and applies optional restrictions.
def training_participants(dataset, family, train_ids=None, ancestry=None, sex=None):
    participants = read_participants(dataset, split="train")

    if train_ids is not None:
        participants = participants.iloc[id_positions(participants, train_ids)]

    # Ancestry and sex filters restrict the stored training split.
    sex_code = {"female": "0", "male": "1"}.get(sex)
    participants = filter_participants(participants, ancestry=[ancestry] if ancestry else None,
                                      sex=[sex_code] if sex_code is not None else None).copy()

    if participants.empty:
        raise ValueError("GWAS requires a nonempty subset of the saved training split.")

    y = participants.phenotype.to_numpy(dtype=float)

    if not np.isfinite(y).all():
        raise ValueError("Training phenotypes must be finite.")

    if family == "binomial" and set(y) != {0.0, 1.0}:
        raise ValueError("Binary GWAS requires both cases (1) and controls (0).")

    if family == "gaussian" and len(np.unique(y)) < 3:
        raise ValueError("Quantitative GWAS requires at least three distinct phenotype values.")

    return participants


# Genotype ID validation requires every selected participant in the chromosome genotype file.
def check_genotype_ids(prefix, participants):
    samples = pd.read_csv(f"{prefix}.psam", sep=r"\s+", dtype=str)
    samples = samples.rename(columns={"#FID": "FID", "#IID": "IID"})

    if "FID" not in samples:
        samples["FID"] = "0"

    ids = samples[["FID", "IID"]]

    if ids.isna().any().any() or ids.duplicated().any():
        raise ValueError("Genotype FID/IID pairs must be complete and unique.")

    available = pd.MultiIndex.from_frame(ids)
    selected = pd.MultiIndex.from_frame(participants[["FID", "IID"]])
    missing = int((~selected.isin(available)).sum())

    if missing:
        raise ValueError(f"{missing} training participants are absent from {prefix}.psam; check ID alignment.")


# Input export preserves stored phenotypes and uses the standard age, sex and PC covariates.
def write_inputs(participants, phenotype, output, num_pcs=10):
    names = ["Sex"] + [f"PC{i}" for i in range(1, num_pcs + 1)] + ["Age"]
    covariates = participants[names].to_numpy(dtype=float)

    if not np.isfinite(covariates).all():
        raise ValueError("Training GWAS covariates must be finite.")

    # Constant covariates, including sex within a sex-specific group, add no information.
    constant = np.ptp(covariates, axis=0) == 0
    dropped = [name for name, remove in zip(names, constant) if remove]
    names = [name for name, remove in zip(names, constant) if not remove]

    if not names:
        raise ValueError("No varying GWAS covariates remain.")

    # PLINK input files contain participant IDs, phenotypes and covariates.
    output.mkdir(parents=True, exist_ok=True)
    participants[["FID", "IID"]].to_csv(output / "keep.tsv", sep="\t", index=False, header=False)
    participants[["FID", "IID", "phenotype"]].rename(columns={"phenotype": phenotype}).to_csv(
        output / "phenotype.tsv", sep="\t", index=False)
    participants[["FID", "IID", *names]].to_csv(output / "covariates.tsv", sep="\t", index=False)

    return names, dropped


# The additive association command uses the same phenotype coding as fitting.
def association_command(plink, genotype, inputs, output, phenotype, family, names, mac, resources):
    command = [plink, "--bfile", str(genotype), "--keep", str(inputs / "keep.tsv"),
               "--pheno", str(inputs / "phenotype.tsv"), "--pheno-name", phenotype,
               "--covar", str(inputs / "covariates.tsv"), "--covar-name", *names,
               "--glm", "hide-covar", "omit-ref", "--mac", str(mac),
               "--out", str(output), *resources]

    if family == "binomial":

        # --1 interprets 0/1 as control/case; PLINK otherwise expects 1/2.
        command += ["--1"]

    # Variance standardization conditions continuous covariates numerically and retains the original sex coding.
    standardized = [name for name in names if name != "Sex"]

    if standardized:
        command += ["--covar-variance-standardize", *standardized]

    return command


# GWAS uses temporary hard-call exports and retains the association results and logs.
def run_gwas(dataset, phenotype, family, chromosome, pfile="data/ukbb/genotypes/chr{chrom}",
             variant_ranges="data/scoring_variants_GRCh37.tsv", analysis_dir="data/analyses/interpretation",
             train_ids=None, ancestry=None, sex=None, num_pcs=10, mac=20, threads=1,
             memory_mb=None, work_dir="data/tmp", plink="plink2"):

    assert 1 <= chromosome <= 22 and num_pcs >= 0 and mac >= 0 and threads >= 1
    assert family in ("gaussian", "binomial") and sex in (None, "female", "male")

    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", phenotype)

    group = "_".join(value for value in (ancestry, sex) if value) or "all"

    assert re.fullmatch(r"[A-Za-z0-9_-]+", group)

    prefix = Path(str(pfile).format(chrom=chromosome))

    # Training participant alignment precedes association input export.
    participants = training_participants(dataset, family, train_ids, ancestry, sex)
    check_genotype_ids(prefix, participants)
    output = Path(analysis_dir) / phenotype / "gwas" / "restricted" / group / f"chr{chromosome}"
    names, dropped = write_inputs(participants, phenotype, output, num_pcs)

    resources = ["--threads", str(threads)]

    if memory_mb is not None:
        resources += ["--memory", str(memory_mb)]

    Path(work_dir).mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="training_gwas_", dir=work_dir) as temporary:
        hard_calls = Path(temporary) / "hard_calls"

        # The explicit position union prevents an unintended full-imputed scan.
        command = [plink, "--pfile", str(prefix), "--chr", str(chromosome),
                   "--extract", "range", str(variant_ranges), "--keep", str(output / "keep.tsv"),
                   "--min-alleles", "2", "--max-alleles", "2", "--set-all-var-ids", "@:#:$r:$a",
                   "--new-id-max-allele-len", "100", "missing", "--make-bed",
                   "--out", str(hard_calls), *resources]

        try:
            subprocess.run(command, check=True)
        finally:

            # The conversion log persists after temporary genotype cleanup.
            log = hard_calls.with_suffix(".log")

            if log.exists():
                (output / "hard_calls.log").write_bytes(log.read_bytes())

        # Additive association tests use the restricted hard-call variants.
        command = association_command(plink, hard_calls, output, output / "gwas", phenotype,
                                      family, names, mac, resources)
        subprocess.run(command, check=True)

    suffix = "logistic.hybrid" if family == "binomial" else "linear"
    result = output / f"gwas.{phenotype}.glm.{suffix}"

    if not result.is_file():
        raise FileNotFoundError(f"PLINK did not write the expected association results: {result}")

    print(f"{phenotype}, {group}, chr{chromosome}: {len(participants)} training participants; {result}")

    if dropped:
        print(f"Constant covariates omitted: {', '.join(dropped)}")

    return result


# Each invocation processes one chromosome independently of other chromosome-array tasks.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phenotype", required=True, help="Analysis label, e.g. LDL_MA.")
    parser.add_argument("--family", required=True, choices=("gaussian", "binomial"))
    parser.add_argument("--chromosome", required=True, type=int)
    parser.add_argument("--pfile", default="data/ukbb/genotypes/chr{chrom}", help="Processed GRCh37 PGEN prefix, optionally containing {chrom}.")
    parser.add_argument("--variant-ranges", type=Path, default=Path("data/scoring_variants_GRCh37.tsv"), help="Stage-2 GRCh37 scoring-position union; always applied.")
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analyses/interpretation"))
    parser.add_argument("--train-ids", type=Path, help="Optional subset of the saved training IDs.")
    parser.add_argument("--ancestry", help="Recorded ancestry label, or non-EUR; omitted means all ancestries.")
    parser.add_argument("--sex", choices=("female", "male"), help="Uses the dataset's female=0, male=1 coding.")
    parser.add_argument("--num-pcs", type=int, default=10)
    parser.add_argument("--mac", type=int, default=20, help="Minimum minor-allele count within the selected group.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, help="PLINK workspace limit; otherwise PLINK chooses its default.")
    parser.add_argument("--work-dir", type=Path, default=Path("data/tmp"))
    parser.add_argument("--plink", default="plink2")
    args = parser.parse_args()
    print("Preparing training-only GWAS inputs.", flush=True)
    run_gwas(**vars(args))


if __name__ == "__main__":
    main()
