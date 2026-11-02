"""Define occupied genomic windows from the variants in reference genotype files.

Positions and window boundaries are 1-based and inclusive. Only windows that
contain reference variants receive an ID; shared UKB/CaG variants are assigned
to these windows in the next script. Outputs tables that serve as useful reference 
for assigning variants to windows, used for both UKB and CaG, using UKB GRch37 pos.
"""

import argparse
import gzip

import pandas as pd


# Converts chromosome labels such as 'chr22' to integers.
def _normalize_chr(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^chr", "", case=False, regex=True).astype(int)


# Expands chromosome lists and ranges, removing duplicates.
def _parse_chr_arg(chromosomes: str) -> list[int]:
    out = []

    for token in chromosomes.split(","):
        token = token.strip()

        if not token:
            continue

        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)

            assert lo <= hi

            out.extend(range(lo, hi + 1))
        else:
            out.append(int(token))

    assert out

    return sorted(set(out))


# Reads coordinates in chunks and counts reference variants in each occupied window.
def make_windows(variant_path: str, window_size_bp: int, variant_format: str) -> pd.DataFrame:
    summaries = []
    opener = gzip.open if variant_path.endswith(".gz") else open

    with opener(variant_path, "rt") as handle:

        # BIM has no header; PVAR starts with metadata followed by a #CHROM header.
        if variant_format == "bim":
            columns = ["CHR", "ID", "CM", "BP", "A1", "A2"]
            chromosome_column, position_column = "CHR", "BP"
        else:
            for line in handle:
                if line.startswith("#CHROM"):
                    columns = line.split()
                    break
            else:
                raise ValueError(f"Missing #CHROM/POS header in {variant_path}")

            chromosome_column, position_column = "#CHROM", "POS"

        reader = pd.read_csv(
            handle, sep=r"\s+", names=columns,
            usecols=[chromosome_column, position_column],
            dtype={chromosome_column: str, position_column: int}, chunksize=500_000)

        for variants in reader:
            variants = variants.rename(columns={chromosome_column: "CHR", position_column: "BP"})
            variants["CHR"] = _normalize_chr(variants["CHR"])

            if (variants["BP"] < 1).any():
                raise ValueError("Variant positions must be positive and 1-based.")

            # Position 1,000,000 belongs to the first 1-Mb window, not the second.
            variants["window_id"] = (variants["BP"] - 1) // window_size_bp
            grouped = variants.groupby(["CHR", "window_id"], sort=True)["BP"]
            summaries.append(grouped.agg(n_snps="count", min_bp_obs="min", max_bp_obs="max"))

    # A window may cross read chunks, so combine its counts and observed position range.
    windows = pd.concat(summaries).groupby(level=["CHR", "window_id"], sort=True)
    windows = windows.agg(n_snps=("n_snps", "sum"),
                          min_bp_obs=("min_bp_obs", "min"), max_bp_obs=("max_bp_obs", "max"))
    windows = windows.reset_index()

    windows["start_bp"] = windows["window_id"] * window_size_bp + 1
    windows["end_bp"] = (windows["window_id"] + 1) * window_size_bp

    return windows[["CHR", "window_id", "start_bp", "end_bp", "n_snps", "min_bp_obs", "max_bp_obs"]]


# Number occupied windows across chromosomes in genomic order.
def make_windows_genomewide(variant_template: str, chromosomes: list[int],
                           window_size_bp: int, variant_format: str) -> pd.DataFrame:

    if "{chr}" not in variant_template:
        raise ValueError("Genome-wide template must include '{chr}' placeholder")

    frames = []

    for chromosome in chromosomes:
        variant_path = variant_template.format(chr=chromosome)
        print(f"chr{chromosome}: defining windows from {variant_path}.", flush=True)
        windows = make_windows(variant_path, window_size_bp, variant_format)
        frames.append(windows.rename(columns={"window_id": "window_id_chr"}))

    windows = pd.concat(frames, ignore_index=True)
    windows = windows.sort_values(["CHR", "window_id_chr"]).reset_index(drop=True)
    windows["window_id_global"] = windows.index.astype(int)

    return windows[["CHR", "window_id_chr", "window_id_global", "start_bp", "end_bp",
                    "n_snps", "min_bp_obs", "max_bp_obs"]]


# Choose one genotype metadata format and save its reference-window table.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)

    inputs.add_argument("--bim", help="BIM file for a single chromosome")
    inputs.add_argument("--pvar", help="PVAR or PVAR.gz file for a single chromosome")
    inputs.add_argument("--bim-template", help="BIM path containing '{chr}'")
    inputs.add_argument("--pvar-template", help="PVAR path containing '{chr}'")

    parser.add_argument("--chromosomes", default="1-22", help="Chromosomes, e.g. '1-22' or '1,2,5-7'")
    parser.add_argument("--out", required=True, help="Output TSV path")
    parser.add_argument("--mb-size", type=int, default=1_000_000, help="Window width in base pairs")
    args = parser.parse_args()

    assert args.mb_size >= 1

    variant_format = "bim" if args.bim or args.bim_template else "pvar"
    template = args.bim_template or args.pvar_template

    if template:
        chromosomes = _parse_chr_arg(args.chromosomes)
        windows = make_windows_genomewide(template, chromosomes, args.mb_size, variant_format)
    else:
        windows = make_windows(args.bim or args.pvar, args.mb_size, variant_format)

    windows.to_csv(args.out, sep="\t", index=False)
    print(f"Saved {len(windows):,} reference windows to {args.out}.", flush=True)


if __name__ == "__main__":
    main()
