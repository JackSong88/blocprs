# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Prepare CARTaGENE native covariates and cluster assignments."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import check_unique_ids, prepare_output_directory


def prepare_covariates(covariates_file, clusters_file, output_dir):
    covariates = pd.read_csv(covariates_file, sep="\t", dtype={"FID": str, "IID": str})
    covariates = covariates.drop(columns=["Age2", "Array"]).dropna()
    check_unique_ids(covariates)
    covariates["Sex"] = np.abs(covariates["Sex"] - 1)
    columns = ["FID", "IID", "Sex"] + [f"PC{i}" for i in range(1, 11)] + ["Age"]
    covariates = covariates[columns]

    # Cluster records use FID as the individual identifier in this cohort input.
    clusters = pd.read_csv(clusters_file, sep=r"\s+", dtype={"FID": str, "cluster": int})
    clusters["IID"] = clusters["FID"]
    clusters = clusters.rename(columns={"cluster": "Cluster"})
    check_unique_ids(clusters)
    output_dir = prepare_output_directory(output_dir)
    covariates.to_csv(output_dir / "covars_cartagene_pcs.txt", sep="\t", header=False, index=False)
    clusters[["FID", "IID", "Cluster"]].to_csv(output_dir / "cluster_assignment.txt",
                                               sep="\t", index=False)
    print(f"Wrote {len(covariates):,} covariate rows and {len(clusters):,} cluster assignments to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariates-file", type=Path, required=True)
    parser.add_argument("--clusters-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_covariates(args.covariates_file, args.clusters_file, args.output_dir)


if __name__ == "__main__":
    main()
