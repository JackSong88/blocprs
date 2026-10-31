"""PGEN dosages in canonical variant and requested participant order."""

from pathlib import Path

import numpy as np
import pandas as pd


# Keep the header while skipping optional PLINK metadata lines.
def _read_metadata(path):
    with Path(path).open() as handle:
        while True:
            position = handle.tell()
            if not handle.readline().startswith("##"):
                handle.seek(position)
                break
        return pd.read_csv(handle, sep=r"\s+", dtype=str, keep_default_na=False)


class PgenDosages:

    # PGEN dosages count ALT; canonical IDs do not depend on PVAR's ID column.
    def __init__(self, prefix):
        import pgenlib

        prefix = str(prefix)
        for suffix in (".pgen", ".pvar", ".psam"):
            if prefix.endswith(suffix):
                prefix = prefix[:-len(suffix)]
                break
        self.prefix = prefix
        pvar = _read_metadata(f"{prefix}.pvar")
        chromosome = pvar["#CHROM"].str.replace(r"^chr", "", regex=True)
        position = pd.to_numeric(pvar["POS"], errors="raise").astype(np.int64)
        ref, alt = pvar["REF"].str.upper(), pvar["ALT"].str.upper()

        if alt.str.contains(",", regex=False).any():
            raise ValueError("MTL-PRS requires biallelic PGEN variants")

        self.variants = pd.DataFrame({
            "chromosome": chromosome,
            "position": position,
            "snp": chromosome + ":" + position.astype(str) + ":" + ref + ":" + alt,
            "ref": ref,
            "alt": alt,
            "variant_index": np.arange(len(pvar), dtype=np.int64),
        })

        if self.variants["snp"].duplicated().any():
            raise ValueError("PVAR contains duplicate canonical variant IDs")

        psam = _read_metadata(f"{prefix}.psam").rename(columns={"#FID": "FID", "#IID": "IID"})
        if "FID" not in psam:
            psam["FID"] = psam["IID"]

        self.ids = psam[["FID", "IID"]].to_numpy(str)
        self.n_samples, self.n_variants = len(psam), len(pvar)
        self._sample_lookup = {tuple(pair): i for i, pair in enumerate(self.ids)}

        if len(self._sample_lookup) != self.n_samples:
            raise ValueError("PSAM contains duplicate FID/IID pairs")

        self._reader = pgenlib.PgenReader(
            f"{prefix}.pgen".encode(), raw_sample_ct=self.n_samples, variant_ct=self.n_variants)
        self._active_subset = None

    # Resolve exact FID/IID pairs without dropping or reordering participants.
    def sample_indices(self, ids):
        ids = np.asarray(ids, dtype=str)

        if ids.ndim != 2 or ids.shape[1] != 2:
            raise ValueError("Participant IDs must have FID and IID columns")

        pairs = [tuple(pair) for pair in ids]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Requested participant IDs contain duplicate FID/IID pairs")

        missing = [pair for pair in pairs if pair not in self._sample_lookup]
        if missing:
            raise ValueError(f"{len(missing)} participants are absent from PSAM: {missing[:5]}")

        return np.asarray([self._sample_lookup[pair] for pair in pairs], dtype=np.int64)

    # Pgenlib requires sorted sample subsets; restore the caller's order afterward.
    def read(self, variant_indices, sample_indices=None):
        variants = np.asarray(variant_indices, dtype=np.int64)
        samples = (np.arange(self.n_samples) if sample_indices is None
                   else np.asarray(sample_indices, dtype=np.int64))

        if variants.ndim != 1 or np.any((variants < 0) | (variants >= self.n_variants)):
            raise IndexError("Variant indices are outside the PVAR range")

        if samples.ndim != 1 or np.any((samples < 0) | (samples >= self.n_samples)):
            raise IndexError("Sample indices are outside the PSAM range")

        if len(np.unique(samples)) != len(samples):
            raise ValueError("PGEN sample subsets cannot contain duplicates")

        if not len(variants) or not len(samples):
            return np.empty((len(samples), len(variants)), dtype=np.float32)

        order = np.argsort(samples, kind="stable")
        subset = np.ascontiguousarray(samples[order], dtype=np.uint32)
        if self._active_subset is None or not np.array_equal(subset, self._active_subset):
            self._reader.change_sample_subset(subset)
            self._active_subset = subset

        dosage = np.empty((len(variants), len(samples)), dtype=np.float32)
        self._reader.read_dosages_list(np.ascontiguousarray(variants, dtype=np.uint32), dosage)

        # Convert Pgenlib's missing-dosage sentinel before later mean imputation.
        dosage[dosage < -8.0] = np.nan
        return np.ascontiguousarray(dosage.T[np.argsort(order)])

    # Release the native reader exactly once.
    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    # Scope each chromosome's reader to its operation.
    def __enter__(self):
        return self

    # Close on both successful operations and exceptions.
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
