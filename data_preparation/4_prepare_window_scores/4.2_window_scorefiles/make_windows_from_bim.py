"""Construct fixed-width, 1-based inclusive windows from reference genotypes."""

import argparse
from collections import defaultdict
import gzip
import pandas as pd


# Open plain or gzip-compressed variant metadata.
def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


# Convert chromosome labels to integers.
def _normalize_chr(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^chr", "", case=False, regex=True).astype(int)


# Expand chromosome lists and ranges, removing duplicates.
def _parse_chr_arg(chromosomes: str) -> list[int]:
    out = []
    for token in chromosomes.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)

            if lo > hi:
                raise ValueError(f"Invalid range '{token}': start > end")

            out.extend(range(lo, hi + 1))
        else:
            out.append(int(token))

    if not out:
        raise ValueError("No chromosomes parsed from --chromosomes")

    return sorted(set(out))


def make_windows_from_bim(bim_path: str, mb_size: int) -> pd.DataFrame:
    """Create occupied windows from BIM metadata.

    n_snps counts reference variant rows in each window.
    """
    variants = pd.read_csv(bim_path, sep=r"\s+", header=None,
                           names=["CHR", "ID", "CM", "BP", "A1", "A2"],
                           dtype={"CHR": str, "ID": str, "BP": int})
    variants["CHR"] = _normalize_chr(variants["CHR"])
    return make_windows_from_variants(variants[["CHR", "BP"]], mb_size)


def make_windows_from_pvar(pvar_path: str, mb_size: int) -> pd.DataFrame:
    """Create occupied windows from plain or gzip-compressed PVAR metadata.

    n_snps counts reference variant rows in each window.
    """
    counts = defaultdict(int)
    min_bp = {}
    max_bp = {}
    chrom_idx = None
    pos_idx = None

    with _open_text(pvar_path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split()
                chrom_idx = header.index("#CHROM")
                pos_idx = header.index("POS")
                continue

            if chrom_idx is None or pos_idx is None:
                raise ValueError(f"Missing #CHROM/POS header in {pvar_path}")

            fields = line.split()
            chrom = int(fields[chrom_idx].removeprefix("chr").removeprefix("CHR"))
            bp = int(fields[pos_idx])

            if bp < 1:
                raise ValueError("Variant positions must be positive and 1-based.")

            window_id = (bp - 1) // mb_size
            key = (chrom, window_id)
            counts[key] += 1
            min_bp[key] = bp if key not in min_bp else min(min_bp[key], bp)
            max_bp[key] = bp if key not in max_bp else max(max_bp[key], bp)

    rows = []
    for chrom, window_id in sorted(counts):
        rows.append({"CHR": chrom, "window_id": window_id,
                     "start_bp": window_id * mb_size + 1, "end_bp": (window_id + 1) * mb_size,
                     "n_snps": counts[(chrom, window_id)],
                     "min_bp_obs": min_bp[(chrom, window_id)],
                     "max_bp_obs": max_bp[(chrom, window_id)]})
    return pd.DataFrame(rows)


def make_windows_from_variants(variants: pd.DataFrame, mb_size: int) -> pd.DataFrame:
    """Create occupied windows from chromosome and position columns.

    n_snps counts reference variant rows in each window.
    """

    # 1-based BP -> 0-based window index
    variants = variants.copy()

    if (variants["BP"] < 1).any():
        raise ValueError("Variant positions must be positive and 1-based.")

    variants["window_id"] = (variants["BP"] - 1) // mb_size

    grp = variants.groupby(["CHR", "window_id"], sort=True)["BP"]
    out = grp.agg(n_snps="count", min_bp_obs="min", max_bp_obs="max").reset_index()

    out["start_bp"] = out["window_id"] * mb_size + 1
    out["end_bp"] = (out["window_id"] + 1) * mb_size

    out = out[["CHR", "window_id", "start_bp", "end_bp", "n_snps", "min_bp_obs", "max_bp_obs"]]
    return out


def make_windows_genomewide(variant_template: str, chromosomes: list[int], mb_size: int,
                           variant_format: str) -> pd.DataFrame:
    """Build chromosome windows and assign unique genome-wide window IDs.

    variant_template includes '{chr}', e.g. data/ukbb/genotypes/chr{chr}.pvar
    for PLINK2 PGEN metadata, or data/ukbb/genotypes/chr{chr}.bim for BED metadata.
    Output columns:
      CHR, window_id_chr, window_id_global, start_bp, end_bp, n_snps, min_bp_obs, max_bp_obs
    n_snps counts reference variant rows in each window.
    """

    if "{chr}" not in variant_template:
        raise ValueError("Genome-wide template must include '{chr}' placeholder")

    frames = []
    for chr_num in chromosomes:
        variant_path = variant_template.format(chr=chr_num)
        if variant_format == "bim":
            df = make_windows_from_bim(variant_path, mb_size)
        elif variant_format == "pvar":
            df = make_windows_from_pvar(variant_path, mb_size)
        else:
            raise ValueError(f"Unsupported variant format: {variant_format}")
        df = df.rename(columns={"window_id": "window_id_chr"})
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["CHR", "window_id_chr"]).reset_index(drop=True)
    out["window_id_global"] = out.index.astype(int)

    out = out[["CHR", "window_id_chr", "window_id_global", "start_bp", "end_bp",
               "n_snps", "min_bp_obs", "max_bp_obs"]]
    return out


# Build windows from the selected reference files and save the table.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bim", help="Path to one chr_*.bim (single-chrom mode)")
    ap.add_argument("--pvar", help="Path to one chr_*.pvar or .pvar.gz (single-chrom mode)")
    ap.add_argument("--bim-template",
                    help="Genome-wide mode template containing '{chr}', e.g. /path/chr_{chr}.bim")
    ap.add_argument("--pvar-template",
                    help="Genome-wide mode template containing '{chr}', e.g. /path/chr_{chr}.pvar")
    ap.add_argument("--chromosomes", default="1-22",
                    help="Comma/range list for genome-wide mode, e.g. '1-22' or '1,2,5-7'")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--mb-size", type=int, default=1_000_000)
    args = ap.parse_args()

    if args.mb_size < 1:
        ap.error("--mb-size must be a positive width in base pairs.")

    input_count = sum(x is not None for x in [args.bim, args.pvar, args.bim_template, args.pvar_template])

    if input_count != 1:
        raise ValueError("Provide exactly one of --bim, --pvar, --bim-template, or --pvar-template")

    if args.bim is not None:
        df = make_windows_from_bim(args.bim, args.mb_size)
    elif args.pvar is not None:
        df = make_windows_from_pvar(args.pvar, args.mb_size)
    else:
        chrs = _parse_chr_arg(args.chromosomes)
        if args.bim_template is not None:
            df = make_windows_genomewide(args.bim_template, chrs, args.mb_size, "bim")
        else:
            df = make_windows_genomewide(args.pvar_template, chrs, args.mb_size, "pvar")

    df.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
