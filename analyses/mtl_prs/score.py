"""Score MTL-PRS candidates, select on validation data, and export SNP effects."""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from blocprs.metrics import IncrementalR2, NagelkerkeR2

from .genotypes import PgenDosages


# Keep participant IDs in their supplied order and reject duplicate pairs.
def _sample_ids(ids):
    ids = np.asarray(ids, dtype=str)

    if ids.ndim != 2 or ids.shape[1] != 2 or len(ids) == 0:
        raise ValueError("Provide participant IDs as a nonempty FID/IID array.")

    if np.any(np.char.strip(ids) == "") or pd.MultiIndex.from_arrays(ids.T).has_duplicates:
        raise ValueError("Participant IDs must be nonempty and unique.")

    return ids


# Resolve one selected candidate or keep the complete candidate matrix.
def _candidate_effects(model, candidate_label):
    labels = model["candidate_labels"].astype(str)
    beta = np.asarray(model["beta_raw"], dtype=np.float64)

    if labels.ndim != 1 or len(labels) == 0 or len(set(labels)) != len(labels):
        raise ValueError("Model candidate labels must be distinct and nonempty.")

    if beta.ndim != 2 or beta.shape[1] != len(labels) or not np.isfinite(beta).all():
        raise ValueError("Model effects must be finite with one column per candidate.")

    if candidate_label is not None:
        matches = np.flatnonzero(labels == candidate_label)

        if len(matches) != 1:
            raise ValueError(f"Candidate {candidate_label!r} is absent from the chromosome model.")

        beta, labels = beta[:, matches], labels[matches]

    return labels, beta


# Score validation or test IDs with ALT-dosage effects and training-mean imputation.
def score_chromosome(genotype_prefix, model_path, ids, output, candidate_label=None,
                     variant_batch_size=512):
    ids = _sample_ids(ids)

    if variant_batch_size < 1:
        raise ValueError("variant_batch_size must be positive.")

    with np.load(model_path, allow_pickle=False) as model, PgenDosages(genotype_prefix) as reader:
        labels, beta = _candidate_effects(model, candidate_label)
        chromosome = int(model["chromosome"].item())
        indices = np.asarray(model["variant_index"], dtype=np.int64)
        means = np.asarray(model["training_mean"], dtype=np.float64)
        train_ids = pd.MultiIndex.from_arrays([model["train_fid"].astype(str), model["train_iid"].astype(str)])

        # Update-training participants cannot serve as validation or test participants.
        if pd.MultiIndex.from_arrays(ids.T).isin(train_ids).any():
            raise ValueError("Scoring IDs overlap the MTL update-training participants.")

        if (indices.shape != (len(beta),) or means.shape != (len(beta),)
            or len(np.unique(indices)) != len(indices) or not np.isfinite(means).all()):
            raise ValueError("Model variant rows, effects and training means are inconsistent.")

        # Saved indices must still refer to the same variants and allele orientation.
        actual = reader.variants.iloc[indices]
        matching = all(np.array_equal(actual[column].to_numpy(dtype=str), model[column].astype(str))
                       for column in ("snp", "ref", "alt", "position"))

        if not matching or not actual.chromosome.astype(int).eq(chromosome).all():
            raise ValueError("Genotype variants or alleles differ from the chromosome model.")

        samples = reader.sample_indices(ids)
        scores = np.zeros((len(ids), len(labels)), dtype=np.float64)

        for start in range(0, len(indices), variant_batch_size):
            rows = slice(start, min(start + variant_batch_size, len(indices)))
            dosage = reader.read(indices[rows], samples).astype(np.float64)

            # Missing genotypes use update-training means, never evaluation means.
            dosage = np.where(np.isfinite(dosage), dosage, means[rows])
            scores += dosage @ beta[rows]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez(handle, chromosome=chromosome, fid=ids[:, 0], iid=ids[:, 1],
                 candidate_labels=labels, scores=scores)
    return output


# Sum a complete chromosome set without reordering participants or candidates.
def combine_scores(paths, ids, chromosomes=range(1, 23)):
    ids = _sample_ids(ids)
    expected = set(map(int, chromosomes))
    seen, labels, total = set(), None, None

    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            chromosome = int(archive["chromosome"].item())

            if chromosome not in expected or chromosome in seen:
                raise ValueError(f"Unexpected or duplicate chromosome score: {chromosome}")

            seen.add(chromosome)
            observed_ids = np.column_stack((archive["fid"].astype(str), archive["iid"].astype(str)))
            current_labels = archive["candidate_labels"].astype(str)
            current_scores = np.asarray(archive["scores"], dtype=np.float64)

            if not np.array_equal(ids, observed_ids):
                raise ValueError(f"Participant IDs or their order differ in {path}")

            if (current_scores.shape != (len(ids), len(current_labels))
                or not np.isfinite(current_scores).all() or len(set(current_labels)) != len(current_labels)):
                raise ValueError(f"Invalid candidate-score matrix in {path}")

            if labels is None:
                labels, total = current_labels.copy(), current_scores.copy()
            else:
                if not np.array_equal(labels, current_labels):
                    raise ValueError(f"Candidate order differs in {path}")

                total += current_scores

    if seen != expected or total is None:
        raise ValueError(f"Incomplete chromosome scores; missing {sorted(expected - seen)}")

    return labels, total


# Candidate zero must reconstruct the calibrated raw source mixture genome-wide.
def check_baseline(scores, source_scores, pi, theta):
    observed = np.asarray(scores, dtype=np.float64)
    expected = np.asarray(source_scores, dtype=np.float64) @ np.asarray(pi, dtype=np.float64) * float(theta)

    if (observed.ndim != 1 or expected.shape != observed.shape or len(observed) < 2
        or not np.isfinite(observed).all() or not np.isfinite(expected).all()):
        raise ValueError("Baseline reconstruction requires aligned finite score vectors.")

    # Allele reversal can add a constant; compare centered predictions.
    observed, expected = observed - observed.mean(), expected - expected.mean()
    observed_ss, expected_ss = np.dot(observed, observed), np.dot(expected, expected)

    if min(observed_ss, expected_ss) <= 0:
        raise ValueError("Baseline reconstruction requires nonconstant scores.")

    difference = observed - expected
    correlation = float(np.dot(observed, expected) / np.sqrt(observed_ss * expected_ss))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    relative_rmse = rmse / np.sqrt(expected_ss / (len(expected) - 1))
    passing = correlation >= .99999 and relative_rmse <= 1e-4

    if not passing:
        raise ValueError(f"Baseline reconstruction failed: correlation={correlation:.8g}, "
                         f"relative_rmse={relative_rmse:.8g}")

    return {"n": len(observed), "correlation": correlation, "rmse_centered": rmse,
            "relative_rmse": float(relative_rmse),
            "slope_centered_observed_on_expected": float(np.dot(observed, expected) / expected_ss),
            "max_absolute_centered_difference": float(np.max(np.abs(difference))), "passing": True}


# Select the best frozen candidate using only the disjoint validation outcomes.
def select_candidate(y, covariates, labels, scores, family):
    labels = np.asarray(labels, dtype=str)
    scores = np.asarray(scores, dtype=np.float64)

    if (scores.shape != (len(y), len(labels)) or len(labels) == 0
        or len(set(labels)) != len(labels) or not np.isfinite(scores).all()):
        raise ValueError("Candidate scores must be finite, aligned and uniquely labeled.")

    metric_type = {"gaussian": IncrementalR2, "binomial": NagelkerkeR2}[family]
    metric = metric_type(y, covariates)
    values = np.asarray([metric.score(scores[:, index]) for index in range(len(labels))])
    selected = int(np.argmax(values))
    return pd.DataFrame({"candidate_label": labels, "candidate_index": np.arange(len(labels)),
                         "metric": "Incremental_R2" if family == "gaussian" else "Nagelkerke_R2",
                         "value": values, "selected": np.arange(len(labels)) == selected})


# Export selected ALT-allele effects and imputation means without genotype data.
def export_weights(model_paths, label, output):
    if label is None:
        raise ValueError("Choose one candidate label for SNP-weight export.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chromosomes, variants = set(), set()

    with gzip.open(output, "wt") as handle:
        for model_path in model_paths:
            with np.load(model_path, allow_pickle=False) as model:
                _, beta = _candidate_effects(model, label)
                chromosome = int(model["chromosome"].item())
                snps = model["snp"].astype(str)

                if chromosome in chromosomes or len(set(snps)) != len(snps) or variants.intersection(snps):
                    raise ValueError("Duplicate chromosome or variant IDs in the selected SNP weights.")

                table = pd.DataFrame({"ID": snps, "effect_allele": model["alt"].astype(str),
                                      "beta": beta[:, 0], "training_mean": model["training_mean"]})
                table.to_csv(handle, sep="\t", index=False, header=not chromosomes)
                chromosomes.add(chromosome)
                variants.update(snps)

    if not chromosomes:
        raise ValueError("No chromosome models supplied for SNP-weight export.")

    return output
