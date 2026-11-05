"""Baseline calibration and individual-level MTL-PRS coordinate paths."""

from pathlib import Path
import logging
from time import monotonic

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    njit = None

from .genotypes import PgenDosages

DEFAULT_FACTORS = (1.0, 10.0, 100.0, 1000.0)
DEFAULT_ITERATIONS = 15


# Baseline calibration regresses standardized y on a centered raw mixture and standardized covariates.
def calibrate_baseline(y, covariates, scores, pi):
    y, covariates = np.asarray(y, dtype=float), np.asarray(covariates, dtype=float)
    scores, pi = np.asarray(scores, dtype=float), np.asarray(pi, dtype=float)

    assert y.ndim == 1 and len(y) >= 2
    assert covariates.ndim == 2 and scores.ndim == 2
    assert len(covariates) == len(y) and len(scores) == len(y)
    assert pi.shape == (scores.shape[1],)

    if not all(np.isfinite(a).all() for a in (y, covariates, scores, pi)):
        raise ValueError("Calibration inputs must be finite")

    # Scaling estimates come from the update-training participants.
    y_mean, y_sd = float(y.mean()), float(y.std(ddof=1))
    covariate_mean = covariates.mean(axis=0)
    covariate_sd = covariates.std(axis=0, ddof=1)

    if y_sd <= 0 or np.any(covariate_sd <= 0):
        raise ValueError("Outcome and covariates must have positive training SD")

    y_standardized = (y - y_mean) / y_sd
    covariates_standardized = (covariates - covariate_mean) / covariate_sd
    initial_score = scores @ pi
    initial_score -= initial_score.mean()

    # Joint OLS estimates the mixture slope and unpenalized covariate effects.
    design = np.column_stack([np.ones(len(y)), initial_score, covariates_standardized])
    coefficients = np.linalg.lstsq(design, y_standardized, rcond=None)[0]

    # The residual excludes fitted covariate effects but retains the genetic signal for SNP updates.
    residual = y_standardized - coefficients[0] - covariates_standardized @ coefficients[2:]

    return {
        "residual": residual,
        "theta": float(coefficients[1]),
        "y_mean": y_mean,
        "y_sd": y_sd,
        "covariate_mean": covariate_mean,
        "covariate_sd": covariate_sd,
        "covariate_coef": coefficients[2:],
        "intercept": float(coefficients[0]),
    }


# Candidate labels identify the baseline and each sweep in factor-major order.
def candidate_labels(factors=DEFAULT_FACTORS, iterations=DEFAULT_ITERATIONS):
    return np.asarray(["baseline"] + [
        f"factor_{factor:g}.iter_{iteration:02d}"
        for factor in factors for iteration in range(1, iterations + 1)
    ])


# Each learning-rate path restarts from the baseline and updates the residual after every coordinate.
def _coordinate_kernel(beta_initial, gram, gy, total_variants, factors, iterations):
    output = np.empty((len(beta_initial), 1 + len(factors) * iterations), dtype=np.float64)
    output[:, 0] = beta_initial
    candidate = 1

    for factor in factors:
        alpha = min(float(factor) / total_variants, 1.0)
        beta = beta_initial.copy()
        residual_gradient = gy - gram @ beta

        for _ in range(iterations):
            for variant in range(len(beta)):
                delta = alpha * residual_gradient[variant]
                beta[variant] += delta

                # The Gram-matrix update adjusts all residual correlations without re-reading participant genotypes.
                residual_gradient -= gram[:, variant] * delta

            # Each complete SNP sweep contributes one candidate.
            output[:, candidate] = beta
            candidate += 1

    return output


if njit is not None:
    _coordinate_kernel = njit(cache=False)(_coordinate_kernel)


# M is the genome-wide nonzero matched weight count before monomorphic filtering.
def coordinate_update_candidates(beta_initial, gram, gy, total_variants,
                                 factors=DEFAULT_FACTORS, iterations=DEFAULT_ITERATIONS):
    beta_initial = np.asarray(beta_initial, dtype=float)
    gram, gy = np.asarray(gram, dtype=float), np.asarray(gy, dtype=float)
    factors = np.asarray(factors, dtype=float)
    m = len(beta_initial)

    assert beta_initial.shape == (m,) and gram.shape == (m, m) and gy.shape == (m,)

    if total_variants <= 0 or iterations < 1 or np.any(factors <= 0):
        raise ValueError("Variant count, iterations, and learning-rate factors must be positive")

    return _coordinate_kernel(beta_initial, gram, gy, int(total_variants), factors, int(iterations))


# Block assignments match upstream end-defined (previous_end, end] bins, including the final tail.
def _block_indices(path, chromosome, positions):
    if str(path).lower() in {"single", "single-block", "none"} or not len(positions):
        return np.zeros(len(positions), dtype=np.int64)

    blocks = pd.read_csv(path, sep=r"\s+", comment="#", header=None, usecols=[0, 1, 2])
    blocks.columns = ["chromosome", "start", "end"]
    block_chromosome = blocks["chromosome"].astype(str).str.replace(r"^chr", "", regex=True)
    ends = pd.to_numeric(blocks.loc[block_chromosome == chromosome, "end"], errors="coerce")
    ends = np.sort(ends.dropna().unique())

    if not len(ends):
        raise ValueError(f"No LD blocks were found for chromosome {chromosome}")

    if np.any(positions < 1):
        raise ValueError("Variant positions must be one-based positive integers")

    return np.searchsorted(ends, positions, side="left").astype(np.int64)


# Dosage SDs reflect mean imputation and use the full training n - 1 denominator.
def _variant_statistics(reader, variants, samples, sample_batch_size):
    sums = np.zeros(len(variants), dtype=float)
    squares = np.zeros(len(variants), dtype=float)
    counts = np.zeros(len(variants), dtype=np.int64)

    for start in range(0, len(samples), sample_batch_size):
        genotype = reader.read(variants, samples[start:start + sample_batch_size]).astype(float)
        sums += np.nansum(genotype, axis=0)
        squares += np.nansum(genotype * genotype, axis=0)
        counts += np.isfinite(genotype).sum(axis=0)

    means = np.divide(sums, counts, out=np.full(len(variants), np.nan), where=counts > 0)
    centered_ss = squares - np.divide(
        sums * sums, counts, out=np.zeros(len(variants)), where=counts > 0)

    # Mean-imputed entries add zero centered SS but remain in the sample-size denominator.
    variance = np.divide(
        centered_ss, len(samples) - 1, out=np.full(len(variants), np.nan), where=counts > 0)

    return means, np.sqrt(np.maximum(variance, 0.0))


# Batched accumulation computes one LD block's standardized Gram matrix without loading all samples.
def _sufficient_statistics(reader, variants, samples, residual, means, sds, sample_batch_size):
    gram = np.zeros((len(variants), len(variants)), dtype=float)
    gy = np.zeros(len(variants), dtype=float)

    for start in range(0, len(samples), sample_batch_size):
        stop = min(start + sample_batch_size, len(samples))
        genotype = reader.read(variants, samples[start:stop]).astype(float)
        standardized = (genotype - means) / sds

        # Zero after centering is imputation with the update-training mean.
        standardized[~np.isfinite(standardized)] = 0.0
        gram += standardized.T @ standardized
        gy += standardized.T @ residual[start:stop]

    return gram / (len(samples) - 1), gy / (len(samples) - 1)


# Chromosome fitting verifies variant identities before using saved indices and fits independent LD blocks.
def fit_chromosome(genotype_prefix, weights_path, ids, calibration, total_variants,
                   ld_blocks, chromosome, output, sample_batch_size=4096):
    chromosome = str(chromosome).removeprefix("chr")
    ids = np.asarray(ids, dtype=str)
    residual = np.asarray(calibration["residual"], dtype=float)
    theta = float(calibration["theta"])

    # The weight table supplies this chromosome's allele-aligned source mixture.
    weights = pd.read_csv(weights_path, sep="\t", dtype={
        "chromosome": str, "snp": str, "ref": str, "alt": str,
        "position": np.int64, "variant_index": np.int64, "beta": float,
    }).sort_values("variant_index").reset_index(drop=True)

    if len(ids) < 2 or residual.shape != (len(ids),) or not np.isfinite(residual).all():
        raise ValueError("Calibration residuals must align with at least two training participants")

    if total_variants <= 0 or sample_batch_size < 1 or not np.isfinite(theta):
        raise ValueError("Positive variant count and batch size, and finite calibration are required")

    if weights["variant_index"].duplicated().any() or weights["snp"].duplicated().any():
        raise ValueError("Chromosome weights contain duplicate variants")

    if not (weights["chromosome"] == chromosome).all() or not np.isfinite(weights["beta"]).all():
        raise ValueError("Weights must have the requested chromosome and finite effects")

    labels = candidate_labels()
    columns = {"snp": str, "ref": str, "alt": str, "variant_index": np.int64,
               "position": np.int64, "training_mean": float, "training_sd": float}
    parts = {name: [] for name in (*columns, "beta_raw")}
    dropped = 0

    with PgenDosages(genotype_prefix) as reader:
        samples = reader.sample_indices(ids)
        indices = weights["variant_index"].to_numpy(np.int64)

        if np.any((indices < 0) | (indices >= reader.n_variants)):
            raise ValueError("Saved weight indices are outside the PVAR range")

        identity = ["chromosome", "position", "snp", "ref", "alt"]
        if not np.array_equal(reader.variants.iloc[indices][identity].to_numpy(), weights[identity].to_numpy()):
            raise ValueError("Saved weight indices do not match PVAR variant IDs and alleles")

        weights["block"] = _block_indices(ld_blocks, chromosome, weights["position"].to_numpy())

        # LD estimates use training genotypes within each block and omit cross-block correlations.
        blocks = weights.groupby("block", sort=True)
        started = last_report = monotonic()
        logging.getLogger(__name__).info("chr%s: fitting %s LD blocks", chromosome, len(blocks))

        for number, (_, block) in enumerate(blocks, start=1):
            variants = block["variant_index"].to_numpy(np.int64)
            means, sds = _variant_statistics(reader, variants, samples, sample_batch_size)
            retained = np.isfinite(sds) & (sds > 0)
            dropped += int((~retained).sum())

            if not retained.any():
                continue

            block, variants = block.loc[retained], variants[retained]
            means, sds = means[retained], sds[retained]
            gram, gy = _sufficient_statistics(
                reader, variants, samples, residual, means, sds, sample_batch_size)

            # Genotype SDs and the calibration slope convert per-allele effects to standardized-genotype coefficients.
            beta_initial = block["beta"].to_numpy(float) * sds * theta
            beta = coordinate_update_candidates(beta_initial, gram, gy, total_variants)

            for name in ("snp", "ref", "alt", "variant_index", "position"):
                parts[name].append(block[name].to_numpy(dtype=columns[name]))

            parts["training_mean"].append(means)
            parts["training_sd"].append(sds)

            # Division by genotype SDs restores per-allele effects on the calibrated phenotype scale.
            parts["beta_raw"].append(beta / sds[:, None])

            now = monotonic()
            if now - last_report >= 30 or number == len(blocks):
                logging.getLogger(__name__).info("chr%s: processed %s/%s LD blocks; elapsed=%.1fs",
                                                 chromosome, number, len(blocks), now - started)
                last_report = now

    # Concatenation joins the fitted blocks, retaining every candidate and its training moments.
    arrays = {}

    for name, dtype in columns.items():
        if parts[name]:
            arrays[name] = np.concatenate(parts[name])
        else:
            arrays[name] = np.empty(0, dtype=dtype)

    if parts["beta_raw"]:
        arrays["beta_raw"] = np.concatenate(parts["beta_raw"])
    else:
        arrays["beta_raw"] = np.empty((0, len(labels)), dtype=float)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, chromosome=np.asarray(chromosome), candidate_labels=labels,
             train_fid=ids[:, 0], train_iid=ids[:, 1], **arrays)

    return {"chromosome": chromosome, "training_n": len(ids), "input_variants": len(weights),
            "fitted_variants": len(arrays["snp"]), "dropped_monomorphic": dropped}
