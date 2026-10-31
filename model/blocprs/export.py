"""SNP-weight export from fitted BLocPRS coefficients.

Fitted genetic coefficients are combined with harmonized source SNP weights
to reconstruct the genetic score, including its additive offset. Inputs must
contain the complete scored panel, with consistent variant identifiers,
alleles and genome build across sources.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

__all__ = ["SNPWeights", "regional_coefficients", "export_snp_weights"]


# Check names are unique and nonempty, without outer spaces, tabs, or newlines.
def _names(values, label: str) -> list[str]:
    names = list(values)

    if not names or any(not isinstance(value, str) for value in names):
        raise ValueError(f"{label} must contain nonempty string names.")

    if any(not value.strip() or value != value.strip() or any(char in value for char in "\t\n\r")
           for value in names):
        raise ValueError(f"{label} contains an empty or malformed name.")

    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names.")

    return names


# Check fitted coefficients and compute each raw window score's multiplier.
def _model_arrays(model):
    try:
        sources = _names(model.source_names_, "source_names_")
        windows = _names(model.window_names_, "window_names_")
        gamma = np.asarray(model.gamma_, dtype=np.float64)
        eta = np.asarray(model.eta_, dtype=np.float64)
        window_mean = np.asarray(model.scaler_.window_mean, dtype=np.float64)
        source_sd = np.asarray(model.scaler_.source_sd, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError("Export requires a fitted BLocPRS model.") from exc

    shape = (len(windows), len(sources))

    if gamma.shape != (len(sources),) or source_sd.shape != gamma.shape:
        raise ValueError("Global coefficients or source scales have invalid shapes.")

    if eta.shape != shape or window_mean.shape != shape:
        raise ValueError("Regional coefficients or window means have invalid shapes.")

    if any(not np.isfinite(array).all() for array in (gamma, eta, window_mean, source_sd)):
        raise ValueError("Model coefficients and scaling moments must be finite.")

    if np.any(source_sd <= 0):
        raise ValueError("Source scales must be strictly positive.")

    # c_eff = gamma + eta_centered; divide by training SD to weight raw scores.
    centered = eta - eta.mean(axis=0, keepdims=True)
    effective = gamma[None, :] + centered
    multipliers = effective / source_sd[None, :]

    if not np.isfinite(multipliers).all():
        raise ValueError("Raw-score multipliers overflowed; check model scales.")

    return sources, windows, gamma, eta, window_mean, centered, effective, multipliers


# Return genetic coefficients in fitted order, excluding intercept and covariates.
# The table contains multipliers; export_snp_weights returns the centering offset.
def regional_coefficients(model) -> pd.DataFrame:
    sources, windows, gamma, eta, _, centered, effective, multipliers = _model_arrays(model)
    return pd.DataFrame({
        "source": np.tile(sources, len(windows)),
        "window": np.repeat(windows, len(sources)),
        "gamma": np.tile(gamma, len(windows)),
        "eta": eta.ravel(),
        "eta_centered": centered.ravel(),
        "c_eff": effective.ravel(),
        "raw_multiplier": multipliers.ravel(),
    })


@dataclass(frozen=True)
class SNPWeights:
    """Combined SNP effect estimates, additive genetic offset and genome build.

    With the original scoring conventions and matching effect-allele dosages,
    ``dosages @ weights.weight + offset`` reconstructs the genetic score up to
    rounding differences. It excludes the model intercept and covariate effects.
    For binary models this is the genetic part of the logit, not a probability.
    The offset includes training centering and any allele-orientation constants.
    """

    weights: pd.DataFrame
    offset: float
    genome_build: str

    # Write a TSV, then replace the destination only after writing finishes.
    # Include the genome build and genetic centering offset in the TSV comments.
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        temporary = None

        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(f"# genome_build={self.genome_build}\n")
                handle.write(f"# genetic_centering_offset={self.offset:.17g}\n")
                handle.write("# predictor=additive_genetic_score_only\n")
                self.weights.to_csv(handle, sep="\t", index=False, float_format="%.17g")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        return path


# Validate text labels; allow_integer also accepts integer chromosome labels.
def _text_column(frame: pd.DataFrame, column: str, *, allow_integer=False) -> None:
    values = frame[column]

    if values.isna().any():
        raise ValueError(f"{column} contains missing values.")

    allowed = (str, int, np.integer) if allow_integer else (str,)

    if any(not isinstance(value, allowed) or isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{column} must contain nonempty strings"
                         + (" or integer chromosome labels." if allow_integer else "."))

    values = values.astype(str)

    if any(not value.strip() or value != value.strip() or any(char in value for char in "\t\n\r")
           for value in values):
        raise ValueError(f"{column} contains an empty or malformed value.")

    frame[column] = values


# Read finite real numbers without treating malformed or missing values as zero.
def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        if any(isinstance(value, (bool, np.bool_, complex, np.complexfloating)) for value in frame[column]):
            raise ValueError("Boolean and complex values are not supported.")

        values = pd.to_numeric(frame[column], errors="raise")

        if values.dtype.kind in "bc":
            raise ValueError("Boolean and complex values are not supported.")

        result = values.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain finite real numbers.") from exc

    if not np.isfinite(result).all():
        raise ValueError(f"{column} contains missing or non-finite values.")

    return result


# Combine the complete, already harmonized source weights into one genetic predictor.
# Requires the dosage and imputation convention used for source scoring.
# Optional allele_offset contains each row's constant in raw source-score units.
def export_snp_weights(model, source_weights: pd.DataFrame, path: str | Path | None = None) -> SNPWeights:
    columns = [
        "source", "window", "variant_id", "chromosome", "position",
        "effect_allele", "other_allele", "weight", "genome_build",
    ]

    if not isinstance(source_weights, pd.DataFrame):
        raise TypeError("source_weights must be a pandas DataFrame.")

    if not source_weights.columns.is_unique:
        raise ValueError("source_weights contains duplicate column names.")

    missing = [column for column in columns if column not in source_weights]

    if missing:
        raise ValueError(f"source_weights is missing required columns: {missing}")

    if source_weights.empty:
        raise ValueError("source_weights must contain the complete matched score panel.")

    extra = ["allele_offset"] if "allele_offset" in source_weights else []
    frame = source_weights.loc[:, columns + extra].copy()

    for column in ("source", "window", "variant_id", "effect_allele", "other_allele", "genome_build"):
        _text_column(frame, column)
    _text_column(frame, "chromosome", allow_integer=True)
    frame["weight"] = _numeric_column(frame, "weight")
    allele_offset = _numeric_column(frame, "allele_offset") if extra else None

    positions = _numeric_column(frame, "position")

    if np.any((positions <= 0) | (positions != np.floor(positions)) | (positions >= 2**53)):
        raise ValueError("position must contain positive, exactly represented integers below 2**53.")

    frame["position"] = positions.astype(np.int64)

    if frame["genome_build"].nunique() != 1:
        raise ValueError("source_weights must use one harmonized genome build.")

    if (frame["effect_allele"] == frame["other_allele"]).any():
        raise ValueError("Effect and other alleles must differ.")

    if frame.duplicated(["source", "variant_id"]).any():
        raise ValueError("Each source/variant_id must appear exactly once.")

    sources, windows, _, _, window_mean, _, _, multipliers = _model_arrays(model)

    if set(frame["source"]) != set(sources):
        raise ValueError("source_weights must represent every fitted source and no unknown sources.")

    if not set(frame["window"]).issubset(windows):
        raise ValueError("source_weights contains windows absent from the fitted model.")

    identity = ["chromosome", "position", "effect_allele", "other_allele", "window"]

    if (frame.groupby("variant_id", sort=False)[identity].nunique() > 1).any().any():
        raise ValueError("A variant has inconsistent coordinates, oriented alleles, or window across sources.")

    # Prevent counting one biological allele pair twice under different IDs.
    variants = frame.drop_duplicates("variant_id").copy()
    variants["_allele_pair"] = [tuple(sorted(pair))
                               for pair in zip(variants["effect_allele"], variants["other_allele"])]

    if variants.duplicated(["chromosome", "position", "_allele_pair"]).any():
        raise ValueError("Multiple variant IDs identify the same genomic allele pair.")

    source_index = {name: index for index, name in enumerate(sources)}
    window_index = {name: index for index, name in enumerate(windows)}
    k = frame["source"].map(source_index).to_numpy(dtype=np.int64)
    w = frame["window"].map(window_index).to_numpy(dtype=np.int64)
    present = np.zeros(window_mean.shape, dtype=bool)
    present[w, k] = True

    # Reject missing source/window pairs with a nonzero training mean.
    if np.any((~present) & (window_mean != 0)):
        raise ValueError("A missing source/window pair has a nonzero saved score mean; the panel is incomplete.")

    # Apply each source/window multiplier before combining sources for the same SNP.
    # sort=False keeps the first-seen variant order rather than sorting IDs.
    frame["weight"] = frame["weight"].to_numpy() * multipliers[w, k]
    combined = frame.groupby("variant_id", sort=False)["weight"].sum()
    output = variants.drop(columns=["source", "weight", "_allele_pair", *extra])
    output["weight"] = output["variant_id"].map(combined)
    output = output[[
        "variant_id", "chromosome", "position", "effect_allele",
        "other_allele", "weight", "genome_build", "window",
    ]].reset_index(drop=True)

    # Center training scores and retain constants introduced by allele reversal.
    # Raw dosage scoring has no centering step, so the saved offset supplies it.
    offset = -float(np.sum(multipliers * window_mean))
    if allele_offset is not None:
        offset += float(np.sum(allele_offset * multipliers[w, k]))

    if (not np.isfinite(frame["weight"].to_numpy()).all()
        or not np.isfinite(output["weight"]).all()
        or not np.isfinite(offset)):
        raise ValueError("Effective weights or centering offset overflowed.")

    result = SNPWeights(output, offset, str(frame["genome_build"].iloc[0]))

    if path is not None:
        result.save(path)

    return result
