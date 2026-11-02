# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Extract ten projected PCs and ancestry assignments from pgsc_calc outputs."""

import argparse
from pathlib import Path

import pandas as pd


def check_ids(frame, label):

    # Missing IDs check.
    if frame[["FID", "IID"]].isna().any().any():
        raise ValueError(f"{label}: missing FID or IID.")

    # Duplicate IDs check.
    if frame.duplicated(["FID", "IID"]).any():
        raise ValueError(f"{label}: duplicate FID/IID pairs.")


def prepare_outputs(paths):
    paths = [Path(path) for path in paths]

    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Output paths must be distinct.")

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def extract_ancestry(popsimilarity, sampleset, covariates, covariates_out, ancestry_out):

    # Selects projected PCs and ancestry probabilities for this cohort.
    pcs = [f"PC{i}" for i in range(1, 11)]
    populations = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]
    probabilities = [f"RF_P_{population}" for population in populations]
    columns = ["sampleset", "FID", "IID"] + pcs + probabilities
    columns += ["MostSimilarPop", "MostSimilarPop_LowConfidence"]
    projected = pd.read_csv(popsimilarity, sep="\t", usecols=columns,
                            dtype={"sampleset": str, "FID": str, "IID": str})
    projected = projected.loc[projected["sampleset"].eq(sampleset)].copy()

    if projected.empty:
        raise ValueError(f"No projected ancestry records found for sampleset {sampleset!r}.")

    check_ids(projected, "Projected ancestry")

    # The first three columns are FID/IID/Sex; Age is last in native-PC files.
    native = pd.read_csv(covariates, sep=r"\s+", header=None, dtype={0: str, 1: str})

    if native.shape[1] < 4:
        raise ValueError("Covariates need FID, IID, Sex and Age; native PCs may appear before Age.")

    native = native.iloc[:, [0, 1, 2, -1]].copy()
    native.columns = ["FID", "IID", "Sex", "Age"]
    check_ids(native, "Native covariates")

    # Replaces native PCs with projected PCs while retaining recorded sex and age.
    covariate_table = native.merge(projected[["FID", "IID"] + pcs], on=["FID", "IID"],
                                   how="inner", validate="one_to_one")

    if covariate_table.empty:
        raise ValueError("No participants are shared by native covariates and projected ancestry.")

    # Low-confidence population assignments are labeled OTH.
    low_confidence = projected["MostSimilarPop_LowConfidence"].astype(str).str.lower()

    if not low_confidence.isin(["true", "false"]).all():
        raise ValueError("MostSimilarPop_LowConfidence must contain True or False.")

    projected["Ancestry"] = projected["MostSimilarPop"].where(low_confidence.eq("false"), "OTH")
    ancestry_table = projected.rename(columns=dict(zip(probabilities, populations)))

    # Saves model covariates separately from ancestry assignments.
    prepare_outputs([covariates_out, ancestry_out])
    covariate_table[["FID", "IID", "Sex"] + pcs + ["Age"]].to_csv(
        covariates_out, sep="\t", header=False, index=False)
    ancestry_table[["FID", "IID"] + populations + ["Ancestry"]].to_csv(
        ancestry_out, sep="\t", index=False)

    return len(covariate_table), len(ancestry_table)

# Genome-wide PRSs used in analysis are scored in 4_prepare_window_scores/
# as SNP/variant set could be different between biobanks
# though the genome-wide PRSs from pgsc_calc can serve as a sanity check.
 # but may differ because of variant coverage.

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--popsimilarity", type=Path,
                        help="Projection output (default: data/{sampleset}/pgsc/{sampleset}/score/{sampleset}_popsimilarity.txt.gz).")
    parser.add_argument("--sampleset", default="ukbb", help="Cohort label (default: ukbb).")
    parser.add_argument("--covariates", type=Path,
                        help="Native covariates (default: data/{sampleset}/covariates.tsv; CARTaGENE uses covariates/covars_cartagene_pcs.txt).")
    parser.add_argument("--covariates-out", type=Path,
                        help="Projected covariates (default: data/{sampleset}/projected_covariates.tsv).")
    parser.add_argument("--ancestry-out", type=Path,
                        help="Ancestry assignments (default: data/{sampleset}/ancestry.tsv).")
    args = parser.parse_args()
    print("Reading projected PCs and ancestry assignments.", flush=True)

    # Resolves the cohort's standard input and output locations.
    cohort_dir = Path("data") / args.sampleset
    score_dir = cohort_dir / "pgsc" / args.sampleset / "score"

    if args.popsimilarity is None:
        args.popsimilarity = score_dir / f"{args.sampleset}_popsimilarity.txt.gz"

    if args.covariates is None:
        native_covariates = "covariates/covars_cartagene_pcs.txt" if args.sampleset == "cartagene" else "covariates.tsv"
        args.covariates = cohort_dir / native_covariates

    if args.covariates_out is None:
        args.covariates_out = cohort_dir / "projected_covariates.tsv"

    if args.ancestry_out is None:
        args.ancestry_out = cohort_dir / "ancestry.tsv"

    try:
        covariate_count, ancestry_count = extract_ancestry(
            args.popsimilarity, args.sampleset, args.covariates, args.covariates_out, args.ancestry_out)
        print(f"Wrote {covariate_count:,} covariate rows and {ancestry_count:,} ancestry rows.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
