"""Format VIPRS effects as GRCh37 scores, optionally lifting positions to GRCh38."""

import argparse
import gzip
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


def prepare_weights(input_file):
    weights = pd.read_csv(input_file, sep="\t")
    columns = {"CHR": "chr_name", "SNP": "rsID", "POS": "chr_position", "A1": "effect_allele", "A2": "other_allele", "BETA": "effect_weight"}

    if weights.empty or weights[list(columns)].isna().any().any():
        raise ValueError("VIPRS weights must be nonempty with complete variant, allele and effect columns.")

    if not np.isfinite(pd.to_numeric(weights["BETA"], errors="raise")).all():
        raise ValueError("VIPRS effect weights must be finite.")

    positions = pd.to_numeric(weights["POS"], errors="raise")

    if not (np.isfinite(positions) & positions.gt(0) & positions.eq(np.floor(positions))).all():
        raise ValueError("VIPRS POS must contain positive integer GRCh37 coordinates.")

    return weights.drop(columns=["PIP", "VAR_BETA"], errors="ignore").rename(columns=columns)


# Map 1-based coordinates; pgsc_calc matches source alleles to the target strand.
def liftover_weights(weights):
    from liftover import get_lifter

    converter = get_lifter("hg19", "hg38", one_based=True)
    kept, coordinates = [], []
    for index, row in enumerate(weights.itertuples(index=False)):
        try:
            matches = set(converter[str(row.chr_name).removeprefix("chr")][int(row.chr_position)])
        except KeyError:
            matches = set()

        # Keep only one unambiguous destination on the autosomes used for scoring.
        if len(matches) != 1:
            continue
        chromosome, position, _ = matches.pop()
        chromosome = str(chromosome).removeprefix("chr")
        if not chromosome.isdigit() or not 1 <= int(chromosome) <= 22:
            continue
        kept.append(index)
        coordinates.append((int(chromosome), int(position)))

    if not kept:
        raise ValueError("No unique autosomal GRCh38 mappings were found.")

    # Preserve source IDs, allele labels and effects for cross-build source matching.
    lifted = weights.iloc[kept].copy()
    lifted["chr_name"], lifted["chr_position"] = zip(*coordinates)
    return lifted


def write_weights(weights, pgs_name, phenotype, output_dir, genome_build="GRCh37"):

    if not pgs_name or Path(pgs_name).name != pgs_name or any(char in pgs_name for char in "\n\r\t"):
        raise ValueError("pgs_name must be a single filename component.")

    if not phenotype or any(char in phenotype for char in "\n\r\t"):
        raise ValueError("phenotype must be a nonempty single-line label.")

    destination = Path(output_dir) / genome_build / f"{pgs_name}.txt.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Custom source IDs are not PGS Catalog accessions.
    metadata = [
        "###PGS CATALOG SCORING FILE - see https://www.pgscatalog.org/downloads/#dl_ftp_scoring for additional information",
        "#format_version=2.0", "##POLYGENIC SCORE (PGS) INFORMATION",
        f"#pgs_id=PGS_{pgs_name}", f"#pgs_name={pgs_name}",
        f"#trait_reported={phenotype}", f"#trait_mapped={phenotype}",
        f"#genome_build={genome_build}",
        f"#variants_number={len(weights)}", "#weight_type=beta",
    ]

    # Replace the score file only after writing the complete gzip.
    with NamedTemporaryFile(dir=destination.parent, prefix=".weights_", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            handle.write("\n".join(metadata) + "\n")
            weights.to_csv(handle, sep="\t", index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="Native GRCh37 VIPRS_EM.fit.gz file.")
    parser.add_argument("--pgs-name", required=True, help="Source name, for example URT_M.")
    parser.add_argument("--phenotype", required=True, help="Phenotype label for the scoring-file header.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lift-over", action="store_true", help="Also write GRCh38 weights using 1-based coordinate conversion.")
    args = parser.parse_args()
    try:
        weights = prepare_weights(args.input_file)
        lifted = liftover_weights(weights) if args.lift_over else None
        destination = write_weights(weights, args.pgs_name, args.phenotype, args.output_dir)
        if lifted is not None:
            lifted_destination = write_weights(lifted, args.pgs_name, args.phenotype, args.output_dir, "GRCh38")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {len(weights)} native GRCh37 weights to {destination}.")
    if lifted is not None:
        print(f"Wrote {len(lifted)} GRCh38 weights to {lifted_destination}; "
              f"omitted {len(weights) - len(lifted)} unmapped, ambiguous or non-autosomal rows.")


if __name__ == "__main__":
    main()
