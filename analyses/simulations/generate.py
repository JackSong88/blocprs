"""Generate regional genetic predictors and outcomes on empirical PRS inputs.

Training participants determine score scaling and regional strength. Held-out
participants retain those transformations and receive independent outcome noise.
The supplied phenotypes are replaced, not used to choose the simulated effects.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import bisect
from scipy.special import expit

from blocprs import PRSData, load_dataset
from blocprs.data import Scaler

from .architectures import Truth, build_truth


PANELS = {"LDL_MA": ("gaussian", 0), "ASTHMA_MA": ("binomial", 1)}


# Replace the phenotype while retaining the raw scores, covariates and split IDs.
def with_outcome(data, y):
    return PRSData(data.window_scores, data.source_names, data.window_names, y=y,
                   covariates=data.covariates, covariate_names=data.covariate_names,
                   sample_ids=data.sample_ids)


class SimulationDesign:
    """Training-standardized scores, read in participant blocks from disk."""

    # Compute training moments without using the original phenotype.
    def __init__(self, data, chromosomes, block_size=8192):
        self.data = data
        self.block_size = block_size
        self.n_samples, self.n_windows, self.n_sources = data.window_scores.shape
        self.chromosomes = np.asarray(chromosomes)
        self.scaler = Scaler.fit(with_outcome(data, np.zeros(data.n_samples)),
                                 "gaussian", block_size)
        self.genomewide = self.global_scores(data)
        self.covariates = self.scaler.transform_covariates(data.covariates).astype(np.float64)

        # Keep only independent nuisance columns when projecting regional scores.
        nuisance = np.column_stack((np.ones(self.n_samples), self.covariates, self.genomewide))
        u, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
        tolerance = np.finfo(float).eps * max(nuisance.shape) * singular[0]
        self.basis = u[:, singular > tolerance]

    # Yield standardized windows without loading the full score matrix into RAM.
    def blocks(self, data):
        for start in range(0, data.n_samples, self.block_size):
            rows = slice(start, min(start + self.block_size, data.n_samples))
            yield rows, self.scaler.transform_windows(data.window_scores[rows])

    # Sum standardized windows using the same training moments in either split.
    def global_scores(self, data):
        values = np.empty((data.n_samples, data.n_sources), dtype=np.float64)

        for rows, scores in self.blocks(data):
            values[rows] = scores.sum(axis=1, dtype=np.float64)

        return values

    # Compute sum_wk(S_iwk * eta_centered_wk), without covariate effects.
    def regional(self, eta_centered, data=None):
        data = self.data if data is None else data
        values = np.empty(data.n_samples, dtype=np.float64)

        for rows, scores in self.blocks(data):
            values[rows] = np.einsum("iwk,wk->i", scores, eta_centered, dtype=np.float64)

        return values

    # Measure regional score variance remaining after all genome-wide scores and C.
    def window_strengths(self):
        total_ss = np.zeros((self.n_windows, self.n_sources))
        projected = np.zeros((self.basis.shape[1], self.n_windows * self.n_sources))

        for rows, scores in self.blocks(self.data):
            flat = scores.reshape(len(scores), -1).astype(np.float64)
            total_ss += np.einsum("ij,ij->j", flat, flat).reshape(total_ss.shape)
            projected += self.basis[rows].T @ flat

        # Orthonormal projection gives residual SS as total SS minus projected SS.
        residual_ss = total_ss - np.sum(projected * projected, axis=0).reshape(total_ss.shape)
        return np.maximum(residual_ss, 0).sum(axis=1) / self.n_samples

    # Load only selected windows for the concentrated-effect calibration.
    def active_scores(self, active):
        scores = np.empty((self.n_samples, len(active), self.n_sources), dtype=np.float32)

        for start in range(0, self.n_samples, self.block_size):
            rows = slice(start, min(start + self.block_size, self.n_samples))
            block = np.array(self.data.window_scores[rows][:, active, :], dtype=np.float32)
            block -= self.scaler.window_mean[active]
            block /= self.scaler.source_sd
            scores[rows] = block

        return scores


@dataclass
class Simulation:
    """One generated dataset and its known genome-wide and regional predictors."""

    train: PRSData
    test: PRSData
    truth: Truth
    family: str
    scaler: Scaler
    train_global: np.ndarray
    test_global: np.ndarray
    train_regional: np.ndarray
    test_regional: np.ndarray
    noise_sd: float
    intercept: float

    # Expose the training moments needed to reconstruct the simulated predictor.
    @property
    def source_sd(self):
        return self.scaler.source_sd

    # Return the means subtracted from each window-source score.
    @property
    def window_mean(self):
        return self.scaler.window_mean


# Add independent Gaussian noise or sample Bernoulli outcomes in each split.
def simulate_outcomes(train_genetic, test_genetic, family, seed=20260722, replicate=0, panel=0):
    streams = np.random.SeedSequence([seed, panel, replicate, 1]).spawn(2)
    train_rng, test_rng = (np.random.default_rng(stream) for stream in streams)

    if family == "gaussian":

        # Target 30% genetic variance using the expected noise variance.
        noise_sd = np.sqrt(np.var(train_genetic) * (1 - .30) / .30)
        train_y = train_genetic + train_rng.normal(0, noise_sd, len(train_genetic))
        test_y = test_genetic + test_rng.normal(0, noise_sd, len(test_genetic))
        return train_y, test_y, float(noise_sd), 0.0

    # Set mean training event probability to 10%; sampled prevalence can differ.
    prevalence = lambda intercept: np.mean(expit(intercept + train_genetic)) - .10

    # These intercept endpoints put all training probabilities near zero or one.
    intercept = bisect(prevalence, -float(np.max(train_genetic)) - 40,
                       -float(np.min(train_genetic)) + 40, xtol=1e-12)
    train_y = train_rng.binomial(1, expit(intercept + train_genetic))
    test_y = test_rng.binomial(1, expit(intercept + test_genetic))
    return train_y, test_y, 0.0, float(intercept)


# Select the dataset's occupied windows without renumbering the reference map.
def read_windows(window_map, window_names):
    windows = pd.read_csv(window_map, sep="\t")
    ids = pd.to_numeric(windows.window_id_global, errors="raise").to_numpy(dtype=float)

    if not np.isfinite(ids).all() or np.any((ids < 0) | (ids != np.floor(ids))):
        raise ValueError("Reference window IDs must be nonnegative integers.")

    windows["window"] = [f"w{int(value):04d}" for value in ids]
    windows = windows.set_index("window", verify_integrity=True).loc[list(window_names)].copy()
    coordinates = windows[["CHR", "start_bp"]].apply(pd.to_numeric, errors="raise")
    values = coordinates.to_numpy(dtype=float)

    if not np.isfinite(values).all() or np.any((values < 1) | (values != np.floor(values))):
        raise ValueError("Window chromosomes and start positions must be positive integers.")

    windows[["CHR", "start_bp"]] = coordinates.astype(np.int64)
    locations = list(zip(windows.CHR, windows.start_bp))

    if locations != sorted(locations) or len(set(locations)) != len(locations):
        raise ValueError("Simulation windows must have distinct, ordered chromosome positions.")

    return windows


# Load the original split, generate the chosen architecture, and replace outcomes.
def prepare_simulation(dataset, window_map, panel, architecture, replicate=0,
                       seed=20260722, block_size=8192):
    family, panel_index = PANELS[panel]
    train = load_dataset(dataset, split="train")
    test = load_dataset(dataset, split="test")
    windows = read_windows(window_map, train.window_names)

    design = SimulationDesign(train, windows.CHR.to_numpy(), block_size)
    truth = build_truth(design, architecture, seed, replicate, panel_index)
    train_global = design.genomewide @ truth.gamma
    train_regional = design.regional(truth.eta_centered)
    test_global = design.global_scores(test) @ truth.gamma
    test_regional = design.regional(truth.eta_centered, test)

    # Keep binary genetic logits at unit training SD, scaling all coefficients together.
    if family == "binomial":
        multiplier = 1 / np.std(train_global + train_regional)
        truth.gamma0 *= multiplier
        truth.gamma *= multiplier
        truth.eta_centered *= multiplier
        truth.delta *= multiplier
        train_global *= multiplier
        train_regional *= multiplier
        test_global *= multiplier
        test_regional *= multiplier

    train_y, test_y, noise_sd, intercept = simulate_outcomes(
        train_global + train_regional, test_global + test_regional,
        family, seed, replicate, panel_index)

    return Simulation(with_outcome(train, train_y), with_outcome(test, test_y), truth,
                       family, design.scaler, train_global, test_global,
                       train_regional, test_regional, noise_sd, intercept)
