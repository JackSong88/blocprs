"""Prepare target-training splits and allele-aligned source-mixture weights."""

from pathlib import Path

import numpy as np
import pandas as pd

from blocprs import load_dataset
from blocprs.cv import _subsample_indices

from .genotypes import PgenDosages


# Load training scores and reserve validation participants without changing the outer split.
def prepare_data(dataset, family, validation_size=50000, seed=42, validation_ids=None):
    dataset = Path(dataset)
    train = load_dataset(dataset, split="train")
    participants = pd.read_csv(dataset / "participants.tsv", sep="\t", dtype={"FID": str, "IID": str})
    test_ids = participants.loc[participants.split.eq("test"), ["FID", "IID"]].to_numpy(dtype=str)

    if not len(test_ids):
        raise ValueError("MTL-PRS requires a stored outer test split.")

    # Sum windows in bounded blocks; genome-wide datasets have one window.
    scores = np.empty((train.n_samples, train.n_sources), dtype=np.float64)
    for start in range(0, train.n_samples, 4096):
        stop = min(start + 4096, train.n_samples)
        scores[start:stop] = np.sum(train.window_scores[start:stop], axis=1, dtype=np.float64)

    y = np.asarray(train.y, dtype=np.float64)
    covariates = np.asarray(train.covariates, dtype=np.float64)

    if not all(np.isfinite(values).all() for values in (y, covariates, scores)):
        raise ValueError("Training phenotypes, covariates and source scores must be finite.")

    if family == "binomial" and not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Binary training outcomes must contain both 0 and 1.")

    if validation_ids is None:

        if not 0 < validation_size < len(y):
            raise ValueError("Validation size must leave participants for both fitting and selection.")

        validation_rows = _subsample_indices(y, validation_size, np.random.RandomState(seed),
                                             stratify=family == "binomial", min_per_class=5)
    else:
        requested = pd.read_csv(validation_ids, sep="\t", dtype=str)[["FID", "IID"]]

        if requested.isna().any().any() or requested.duplicated().any():
            raise ValueError("Validation IDs must be complete and unique.")

        available = pd.MultiIndex.from_arrays(train.sample_ids.T)
        validation_rows = available.get_indexer(pd.MultiIndex.from_frame(requested))

        if np.any(validation_rows < 0) or not 0 < len(validation_rows) < len(y):
            raise ValueError("Validation IDs must be a nonempty proper subset of outer training.")

    # Preserve dataset row order for both inner partitions.
    validation = np.zeros(len(y), dtype=bool)
    validation[validation_rows] = True
    return {"fid": train.sample_ids[:, 0].astype(str), "iid": train.sample_ids[:, 1].astype(str),
            "test_fid": test_ids[:, 0], "test_iid": test_ids[:, 1], "y": y,
            "covariates": covariates, "scores": scores,
            "source_names": np.asarray(train.source_names), "covariate_names": np.asarray(train.covariate_names),
            "update_rows": np.flatnonzero(~validation), "validation_rows": np.flatnonzero(validation),
            "family": np.asarray(family), "seed": np.asarray(seed), "dataset": np.asarray(str(dataset.resolve()))}


# Match accession columns while allowing a source to be absent from one additive shard.
def source_columns(header, sources):
    columns = {}
    for source in sources:
        matches = [name for name in (source, f"{source}_hmPOS_GRCh37", f"{source}_hmPOS_GRCh38") if name in header]

        if len(matches) > 1:
            raise ValueError(f"Ambiguous source columns for {source}: {matches}")

        if matches:
            columns[source] = matches[0]
    return columns


# Combine canonical additive shards, then express all effects on the PGEN ALT allele.
def prepare_weights(chromosome, genotype_prefix, scorefile_template, sources, pi, output):
    if np.asarray(pi).shape != (len(sources),) or not np.isfinite(pi).all():
        raise ValueError("Each source needs one finite mixture coefficient.")

    substitutions = (0, 1) if "{additive}" in scorefile_template else (None,)
    paths = [Path(scorefile_template.format(chrom=chromosome, additive=value)) for value in substitutions]
    paths = [path for path in paths if path.exists()]

    if not paths:
        raise FileNotFoundError(f"No canonical scorefiles for chromosome {chromosome}.")

    frames, found = [], set()
    for path in paths:
        header = pd.read_csv(path, sep="\t", nrows=0).columns
        columns = source_columns(header, sources)
        found.update(columns)
        usecols = ["ID", "effect_allele", *columns.values()]
        for chunk in pd.read_csv(path, sep="\t", usecols=usecols, dtype={"ID": str, "effect_allele": str}, chunksize=100000):
            beta = np.zeros(len(chunk), dtype=np.float64)
            for source, coefficient in zip(sources, pi):
                if source not in columns:
                    continue

                # Missing sparse weights are zero; malformed numeric values are errors.
                values = pd.to_numeric(chunk[columns[source]], errors="raise").fillna(0).to_numpy(dtype=float)

                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite source weights in {path}: {source}")

                beta += coefficient * values

            selected = beta != 0
            frames.append(pd.DataFrame({"snp": chunk.loc[selected, "ID"],
                                        "effect_allele": chunk.loc[selected, "effect_allele"].str.upper(),
                                        "beta": beta[selected]}))

    if set(sources) - found:
        raise ValueError(f"Source columns absent from all shards: {sorted(set(sources) - found)}")

    weights = pd.concat(frames, ignore_index=True)

    if weights[["snp", "effect_allele"]].isna().any().any():
        raise ValueError("Nonzero source weights require SNP IDs and effect alleles.")

    weights = weights.groupby(["snp", "effect_allele"], as_index=False, sort=False).beta.sum()
    weights = weights.loc[weights.beta.ne(0)]
    with PgenDosages(genotype_prefix) as genotypes:
        variants = genotypes.variants
        aligned = weights.merge(variants, on="snp", how="left", validate="many_to_one", indicator=True)

    if not aligned["_merge"].eq("both").all():
        raise ValueError("Canonical nonzero SNP weights are missing from the target genotype file.")

    same = aligned.effect_allele.eq(aligned.alt)
    opposite = aligned.effect_allele.eq(aligned.ref)

    if not (same | opposite).all():
        raise ValueError("Canonical SNP effect alleles do not match the target REF/ALT alleles.")

    # REF-coded effects become negative ALT effects; the resulting constant is absorbed by the intercept.
    aligned["beta"] = np.where(same, aligned.beta, -aligned.beta)
    names = ["chromosome", "position", "snp", "ref", "alt", "variant_index"]
    weights = aligned.groupby(names, as_index=False, sort=False).beta.sum()

    # Only nonzero allele-oriented mixtures enter M and the update paths.
    weights = weights.loc[weights.beta.ne(0)].sort_values("variant_index")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output, sep="\t", index=False, compression="infer")
    return len(weights)
