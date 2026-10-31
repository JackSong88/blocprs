"""Primary A0/A1/A2 truth in the empirical source-window score coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Truth:
    architecture: str
    gamma0: np.ndarray
    gamma: np.ndarray
    eta_centered: np.ndarray
    delta: np.ndarray
    active: np.ndarray
    lambda_sim: float
    regional_strength: float


class InfeasibleTruthError(RuntimeError):
    """The sampled architecture cannot attain its declared strength and bounds."""


# Reuse independent nuisance columns, or build a rank-aware SVD basis.
def _nuisance_basis(design):

    if hasattr(design, "basis"):
        return design.basis

    columns = [np.ones(design.genomewide.shape[0]), design.covariates, design.genomewide]
    matrix = np.column_stack(columns)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular[0]
    return left[:, singular > tolerance]


# Remove covariates and every genome-wide source score before measuring variance.
def _unique_variance(values, basis):
    residual = values - basis @ (basis.T @ values)
    return float(np.var(residual))


# Preserve chromosome quotas while sampling a weighted nonadjacent window set.
def _active_windows(chromosomes, strengths, rng, fraction=0.05):
    """Chromosome quotas and a weighted, exact-size nonadjacent selection."""
    labels = np.asarray(chromosomes).astype(str)
    strengths = np.asarray(strengths, dtype=float)

    if strengths.shape != labels.shape or np.any(~np.isfinite(strengths)):
        raise ValueError("Window strengths must be finite and match chromosomes")

    if np.any(strengths < 0.0) or not np.any(strengths > 0.0):
        raise InfeasibleTruthError("No positive residual window-score variance")

    n_active = min(labels.size, max(2, round(labels.size * fraction)))
    chromosomes, capacities = np.unique(labels, return_counts=True)
    quotas = n_active * capacities / capacities.sum()
    counts = np.floor(quotas).astype(int)

    # Assign leftover windows by fractional quota, with random tie breaks.
    ties = {chrom: rank for rank, chrom in enumerate(rng.permutation(chromosomes))}
    priority = sorted(
        range(chromosomes.size),
        key=lambda j: (quotas[j] - counts[j], -ties[chromosomes[j]]),
        reverse=True,
    )
    for j in priority[: n_active - counts.sum()]:
        counts[j] += 1

    selected = []
    for chrom, count in zip(chromosomes, counts):
        candidates = np.flatnonzero(labels == chrom)
        n = candidates.size

        if count > (n + 1) // 2:
            raise InfeasibleTruthError("A chromosome quota exceeds nonadjacent capacity")

        if count == 0:
            continue

        weights = strengths[candidates] / strengths.mean()
        utility = np.log(np.maximum(weights, np.finfo(float).tiny))
        utility += rng.gumbel(size=n)

        # Maximize the sampled utilities while keeping exactly count nonadjacent windows.
        scores = np.full((n + 1, count + 1), -np.inf)
        take = np.zeros((n + 1, count + 1), dtype=bool)
        scores[:, 0] = 0.0
        for used in range(1, n + 1):
            for size in range(1, min(count, (used + 1) // 2) + 1):
                skip_score = scores[used - 1, size]
                take_score = scores[max(0, used - 2), size - 1] + utility[used - 1]
                take[used, size] = take_score > skip_score
                scores[used, size] = max(skip_score, take_score)
        used, size = n, count
        while size:
            if take[used, size]:
                selected.append(candidates[used - 1])
                used, size = max(0, used - 2), size - 1
            else:
                used -= 1
    return np.asarray(sorted(selected), dtype=int)


# Sample the conditional Dirichlet law without repairing or projecting draws.
def _capacity_dirichlet(upper, rng, max_draws=100000):
    """Dirichlet(2) conditional on recipient capacities, without projection."""
    if upper.sum() < 1.0 - 1e-12:
        raise InfeasibleTruthError("Recipients cannot absorb the maximum transfer")

    if upper.size == 1:
        return np.ones(1)

    if upper.size == 2:
        lower, higher = max(0.0, 1.0 - upper[1]), min(1.0, upper[0])

        if lower > higher + 1e-12:
            raise InfeasibleTruthError("The recipient capacity interval is empty")

        # Two Dirichlet(2) shares reduce to a Beta(2, 2) draw within these bounds.
        cdf = lambda x: 3.0 * x * x - 2.0 * x * x * x
        quantile = cdf(lower) + rng.random() * (cdf(higher) - cdf(lower))
        for _ in range(60):
            middle = (lower + higher) / 2.0
            if cdf(middle) < quantile:
                lower = middle
            else:
                higher = middle
        first = (lower + higher) / 2.0
        return np.array([first, 1.0 - first])

    for _ in range(max_draws):
        weights = rng.dirichlet(np.full(upper.size, 2.0))
        if np.all(weights <= upper + 1e-15):
            return weights
    raise InfeasibleTruthError("Conditional Dirichlet exceeded 100000 internal draws")


# Use the shortest random recipient prefix able to absorb a 95% donor transfer.
def _redistribution(gamma, identity_rng, allocation_rng):
    donor = int(identity_rng.choice(gamma.size, p=gamma / gamma.sum()))
    remaining = np.delete(np.arange(gamma.size), donor)
    order = identity_rng.choice(remaining, size=min(3, remaining.size), replace=False)

    # A 20-fold coefficient cap leaves room for an additional 19 times its baseline.
    capacities = 19.0 * gamma[order] / (0.95 * gamma[donor])
    feasible = np.flatnonzero(np.cumsum(capacities) >= 1.0 - 1e-12)

    if not feasible.size:
        raise InfeasibleTruthError("The sampled recipients lack redistribution capacity")

    count = feasible[0] + 1
    allocation = np.zeros(gamma.size)
    allocation[donor] = -1.0
    allocation[order[:count]] = _capacity_dirichlet(capacities[:count], allocation_rng)
    return gamma[donor] * allocation


# Build independent random streams and literal amplification/attenuation/transfer terms.
def _concentrated_atoms(gamma, n_active, rng):
    """Fixed random blueprint; rows are elementary coefficient perturbations."""
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=5, dtype=np.uint32)
    pattern_rng, count_rng, identity_rng, magnitude_rng, allocation_rng = (
        np.random.default_rng(int(seed)) for seed in seeds
    )
    counts = np.full(3, n_active // 3, dtype=int)
    remainder = n_active % 3

    if remainder:
        counts[pattern_rng.permutation(3)[:remainder]] += 1

    patterns = np.repeat(["amplify", "redistribute", "attenuate"], counts)
    pattern_rng.shuffle(patterns)
    magnitudes = 0.25 + 0.50 * magnitude_rng.beta(2.0, 2.0, n_active)
    atoms, rates, limits = [], [], []
    for window, (pattern, magnitude) in enumerate(zip(patterns, magnitudes)):

        # Draw both allocations so their randomness is independent of mechanism.
        count = int(count_rng.integers(1, min(3, gamma.size) + 1))
        sources = identity_rng.choice(gamma.size, size=count, replace=False)
        shares = allocation_rng.dirichlet(np.full(count, 2.0))
        redistribution = _redistribution(gamma, identity_rng, allocation_rng)
        if pattern == "redistribute":
            atom = np.zeros((n_active, gamma.size))
            atom[window] = redistribution
            atoms.append(atom)
            rates.append(magnitude)
            limits.append(0.95)
        else:
            sign = 1.0 if pattern == "amplify" else -1.0
            for source, share in zip(sources, shares):
                atom = np.zeros((n_active, gamma.size))
                atom[window, source] = sign * gamma[source]
                atoms.append(atom)
                rates.append(magnitude * share)
                limits.append(np.inf if sign > 0.0 else 0.95)
    return np.asarray(atoms), np.asarray(rates), np.asarray(limits)


# Measure regional variation not explained by covariates or genome-wide scores.
def _atomic_covariance(scores, atoms, nuisance_basis, block_size=8192):
    """Residualize in two passes without storing N x atomic-term predictors."""

    # Most perturbation coefficients are zero, so use sparse multiplication.
    matrix = csr_matrix(atoms.reshape(atoms.shape[0], -1))
    projected = np.zeros((nuisance_basis.shape[1], atoms.shape[0]))
    n = scores.shape[0]

    # First accumulate the projection onto genome-wide scores and covariates.
    # With orthonormal columns Q, Q.T @ X gives the projection coefficients.
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        values = (matrix @ scores[start:stop].reshape(stop - start, -1).T).T
        projected += nuisance_basis[start:stop].T @ values

    # Re-read blocks to subtract the projection without storing all participant scores.
    # This trades a second pass through the scores for lower memory use.
    covariance = np.zeros((atoms.shape[0], atoms.shape[0]))
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        residual = (matrix @ scores[start:stop].reshape(stop - start, -1).T).T
        residual -= nuisance_basis[start:stop] @ projected
        covariance += residual.T @ residual

    # The nuisance basis includes an intercept, so the residuals are already centered.
    return covariance / n


# Ignore a recipient cap that its already-saturated donor can never reach.
def _lambda_upper(atoms, rates, limits, gamma0):
    """Only caps reached before an atom saturates restrict the common lambda."""
    bounds = []
    for atom, rate, limit in zip(atoms, rates, limits):
        positive = atom > 0.0

        if not np.any(positive):
            continue

        allowance = np.broadcast_to(19.0 * gamma0, atom.shape)
        allowed_weight = float(np.min(allowance[positive] / atom[positive]))

        # A recipient safely below its cap after donor saturation adds no bound.
        if not np.isfinite(limit) or allowed_weight < limit - 1e-12:
            bounds.append(allowed_weight / rate)
    if bounds:
        return float(min(bounds))

    if np.all(np.isfinite(limits)):
        return float(np.max(limits / rates))

    raise InfeasibleTruthError("No finite coefficient guardrail for the lambda path")


# Use cancellation-resistant quadratic roots, including the linear limit.
def _quadratic_roots(coefficients):
    quadratic, linear, constant = coefficients
    tolerance = 64.0 * np.finfo(float).eps * np.max(np.abs(coefficients))

    if abs(quadratic) <= tolerance:
        return [] if abs(linear) <= tolerance else [-constant / linear]

    discriminant = linear * linear - 4.0 * quadratic * constant
    error = 64.0 * np.finfo(float).eps * (
        linear * linear + abs(4.0 * quadratic * constant)
    )
    if discriminant < -error:
        return []

    numerator = -0.5 * (linear + np.copysign(np.sqrt(max(discriminant, 0.0)), linear))
    if numerator == 0.0:
        return [-linear / (2.0 * quadratic)]

    return [numerator / quadratic, constant / numerator]


# Search all saturation intervals and retain the first feasible target crossing.
def _smallest_lambda(covariance, genomewide_covariance, gamma0, shifts,
                     rates, limits, upper, target=0.10):
    """Find the first crossing of the piecewise-quadratic variance ratio."""

    # Each breakpoint is the multiplier at which one effect term reaches its cap.
    breakpoints = limits / rates
    internal = breakpoints[(breakpoints > 0.0) & (breakpoints < upper)]
    boundaries = np.unique(np.concatenate([[0.0, upper], internal]))

    # Evaluate the actual clipped path when validating a candidate root.
    def ratio(strength):
        weights = np.minimum(strength * rates, limits)
        gamma = gamma0 + shifts.T @ weights
        return float(weights @ covariance @ weights) / float(
            gamma @ genomewide_covariance @ gamma
        )

    candidates = [point for point in boundaries
                  if np.isclose(ratio(point), target, rtol=1e-9, atol=1e-12)]
    for lower, higher in zip(boundaries[:-1], boundaries[1:]):

        # Within this interval, weights = intercept + strength * slope.
        saturated = 0.5 * (lower + higher) * rates >= limits
        intercept = np.where(saturated, limits, 0.0)
        slope = np.where(saturated, 0.0, rates)
        gamma_intercept = gamma0 + shifts.T @ intercept
        gamma_slope = shifts.T @ slope
        numerator = np.array([
            slope @ covariance @ slope,
            2.0 * intercept @ covariance @ slope,
            intercept @ covariance @ intercept,
        ])
        denominator = np.array([
            gamma_slope @ genomewide_covariance @ gamma_slope,
            2.0 * gamma_intercept @ genomewide_covariance @ gamma_slope,
            gamma_intercept @ genomewide_covariance @ gamma_intercept,
        ])

        # Solve regional_variance - target * genomewide_variance = 0.
        for root in _quadratic_roots(numerator - target * denominator):
            tolerance = 1e-10 * max(1.0, abs(lower), abs(higher))
            if np.isfinite(root) and lower - tolerance <= root <= higher + tolerance:
                root = float(np.clip(root, lower, higher))
                if np.isclose(ratio(root), target, rtol=1e-8, atol=1e-12):
                    candidates.append(root)

    if not candidates:
        raise InfeasibleTruthError(
            f"No lambda reaches regional strength {target:g} within the coefficient bounds"
        )

    # Choose the smallest multiplier reaching the target, not a variance minimum.
    return float(min(candidates))


# Draw chromosome-wise AR(1) departures; condition only on calibrated positivity.
def _diffuse(design, gamma, rng, basis):
    chromosomes = np.asarray(design.chromosomes).astype(str)
    target, phi = 0.02, 0.8
    anchor_variance = float(np.var(design.genomewide @ gamma))
    for _ in range(100):
        direction = np.zeros((design.n_windows, design.n_sources))
        for source in range(design.n_sources):
            for chromosome in sorted(set(chromosomes)):
                indices = np.flatnonzero(chromosomes == chromosome)
                innovations = rng.normal(size=indices.size)
                direction[indices[0], source] = innovations[0]

                # Restart each chromosome; the innovation scale preserves unit AR(1) variance.
                for position in range(1, indices.size):
                    direction[indices[position], source] = (
                        phi * direction[indices[position - 1], source]
                        + np.sqrt(1.0 - phi * phi) * innovations[position]
                    )
        direction -= direction.mean(axis=0, keepdims=True)
        direction /= max(float(np.sqrt(np.mean(direction * direction))), 1e-12)
        direction *= gamma
        variance = _unique_variance(design.regional(direction), basis)

        if variance <= 0.0:
            continue

        scale = np.sqrt(target * anchor_variance / variance)
        eta = scale * direction

        if np.all(gamma + eta >= 1e-10):
            return eta, float(scale)

    raise InfeasibleTruthError("No positive A1 architecture after 100 direction draws")


# Generate A0/A1/A2 from paired, outcome-independent random streams.
def build_truth(design, architecture, seed=20260722, replicate=0, panel=0):
    """Generate primary truth without using outcomes or held-out observations.

    The design supplies training genomewide/covariate arrays, chromosome labels,
    regional(eta), window_strengths(), and active_scores(indices). A cached
    rank-aware orthonormal nuisance basis may be provided as design.basis.
    """
    if architecture not in {"A0", "A1", "A2"}:
        raise ValueError("Primary architectures are A0, A1, and A2")

    if design.n_sources < 2 or design.n_windows < 2:
        raise ValueError("Simulation requires at least two sources and windows")

    streams = np.random.SeedSequence([seed, panel, replicate]).spawn(4)
    gamma_rng, diffuse_rng, active_rng, concentrated_rng = (
        np.random.default_rng(stream) for stream in streams
    )
    gamma0 = gamma_rng.dirichlet(np.full(design.n_sources, 2.0))
    initial_variance = float(np.var(design.genomewide @ gamma0))

    if not np.isfinite(initial_variance) or initial_variance <= 0.0:
        raise InfeasibleTruthError("The initial genomewide mixture has no variance")

    gamma0 /= np.sqrt(initial_variance)
    gamma = gamma0.copy()
    eta = np.zeros((design.n_windows, design.n_sources))
    delta = np.zeros_like(eta)
    active = np.empty(0, dtype=int)
    strength = 0.0
    basis = _nuisance_basis(design)

    if architecture == "A1":
        eta, strength = _diffuse(design, gamma0, diffuse_rng, basis)
        delta = eta.copy()
        active = np.arange(design.n_windows)
    elif architecture == "A2":
        active = _active_windows(design.chromosomes, design.window_strengths(), active_rng)
        atoms, rates, limits = _concentrated_atoms(gamma0, active.size, concentrated_rng)
        scores = design.active_scores(active)
        covariance = _atomic_covariance(scores, atoms, basis)
        centered_genomewide = design.genomewide - design.genomewide.mean(axis=0)
        genomewide_covariance = centered_genomewide.T @ centered_genomewide / scores.shape[0]
        shifts = atoms.sum(axis=1) / design.n_windows
        upper = _lambda_upper(atoms, rates, limits, gamma0)
        strength = _smallest_lambda(covariance, genomewide_covariance, gamma0,
                                   shifts, rates, limits, upper)
        weights = np.minimum(strength * rates, limits)
        delta[active] = np.einsum("m,mwk->wk", weights, atoms)

        # Move the mean perturbation into gamma, preserving gamma + eta = gamma0 + delta.
        gamma = gamma0 + delta.mean(axis=0)
        eta = delta - delta.mean(axis=0, keepdims=True)
        effective_ratio = (gamma0 + delta) / gamma0

        if np.min(effective_ratio) < 0.05 - 1e-10 or np.max(effective_ratio) > 20.0 + 1e-10:
            raise InfeasibleTruthError("Final A2 coefficients violate their declared bounds")

    achieved = 0.0 if architecture == "A0" else _unique_variance(
        design.regional(eta), basis
    ) / float(np.var(design.genomewide @ gamma))
    target = {"A0": 0.0, "A1": 0.02, "A2": 0.10}[architecture]

    if not np.isclose(achieved, target, rtol=1e-6, atol=1e-10):
        raise InfeasibleTruthError(f"Final regional strength {achieved:g} misses {target:g}")

    if not np.allclose(eta.sum(axis=0), 0.0, atol=1e-9):
        raise InfeasibleTruthError("Regional coefficients are not centered")

    if not np.allclose(gamma + eta, gamma0 + delta, rtol=1e-12, atol=1e-12):
        raise InfeasibleTruthError("Effective coefficient reconstruction failed")

    return Truth(architecture, gamma0, gamma, eta, delta, active, strength, achieved)
