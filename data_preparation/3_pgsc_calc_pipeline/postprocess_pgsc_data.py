# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Extract raw SUM scores or projected covariates from pgsc_calc outputs."""

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


def extract_scores(scores_file, sampleset, source_table, output_dir, analyses=None):
    scores = pd.read_csv(scores_file, sep="\t", usecols=["sampleset", "FID", "IID", "PGS", "SUM"],
                         dtype={"sampleset": str, "FID": str, "IID": str, "PGS": str, "SUM": float})
    scores = scores.loc[scores["sampleset"].eq(sampleset)]

    if scores.empty:
        raise ValueError(f"No scores found for sampleset {sampleset!r}.")

    if scores[["FID", "IID", "PGS"]].isna().any().any():
        raise ValueError("Scores contain missing participant or source IDs.")

    scores = scores.pivot(index=["FID", "IID"], columns="PGS", values="SUM").reset_index()
    scores.columns = [column.replace("_hmPOS_GRCh37", "").replace("_hmPOS_GRCh38", "")
                      for column in scores.columns]

    if scores.columns.duplicated().any():
        raise ValueError("Source names are duplicated after removing build suffixes.")

    panel = pd.read_csv(source_table, dtype=str)
    group = "AnalysisID" if "AnalysisID" in panel.columns else "Phenotype_short"
    if analyses:
        missing = set(analyses).difference(panel[group].dropna())

        if missing:
            raise ValueError(f"Analyses not found in the source table: {sorted(missing)}")

        panel = panel.loc[panel[group].isin(analyses)]

    if panel.empty or panel[[group, "PGS"]].isna().any().any():
        raise ValueError("The selected source table is empty or contains missing analysis/source IDs.")

    outputs = []

    # Check all source panels before writing outputs.
    for analysis, rows in panel.groupby(group):
        sources = list(dict.fromkeys(rows["PGS"]))
        missing = set(sources).difference(scores.columns)

        if missing:
            raise ValueError(f"{analysis}: requested sources are missing from the scores: {sorted(missing)}")

        output = Path(output_dir) / analysis / f"{sampleset}.csv.gz"
        outputs.append((output, sources))

    prepare_outputs([output for output, _ in outputs])
    for output, sources in outputs:
        scores[["FID", "IID"] + sources].to_csv(output, sep="\t", index=False)
    return len(outputs), len(scores)


def extract_ancestry(popsimilarity, sampleset, covariates, covariates_out, ancestry_out):
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

    prepare_outputs([covariates_out, ancestry_out])
    covariate_table[["FID", "IID", "Sex"] + pcs + ["Age"]].to_csv(
        covariates_out, sep="\t", header=False, index=False)
    ancestry_table[["FID", "IID"] + populations + ["Ancestry"]].to_csv(
        ancestry_out, sep="\t", index=False)
    return len(covariate_table), len(ancestry_table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    scores = commands.add_parser("scores", help="Extract per-analysis raw SUM scores.")
    scores.add_argument("--scores-file", type=Path,
                        help="Raw scores (default: data/{sampleset}/pgsc/{sampleset}/score/{sampleset}_pgs.txt.gz).")
    scores.add_argument("--sampleset", default="ukbb", help="Cohort label (default: ukbb).")
    scores.add_argument("--source-table", type=Path, default=Path("data/source_panel.csv"),
                        help="Source-panel CSV (default: data/source_panel.csv).")
    scores.add_argument("--analyses", nargs="+")
    scores.add_argument("--output-dir", type=Path,
                        help="Per-analysis output root (default: data/{sampleset}/source_scores).")

    ancestry = commands.add_parser("ancestry", help="Extract ten projected PCs and ancestry assignments.")
    ancestry.add_argument("--popsimilarity", type=Path,
                          help="Projection output (default: data/{sampleset}/pgsc/{sampleset}/score/{sampleset}_popsimilarity.txt.gz).")
    ancestry.add_argument("--sampleset", default="ukbb", help="Cohort label (default: ukbb).")
    ancestry.add_argument("--covariates", type=Path,
                          help="Native covariates (default: data/{sampleset}/covariates.tsv; CARTaGENE uses covariates/covars_cartagene_pcs.txt).")
    ancestry.add_argument("--covariates-out", type=Path,
                          help="Projected covariates (default: data/{sampleset}/projected_covariates.tsv).")
    ancestry.add_argument("--ancestry-out", type=Path,
                          help="Ancestry assignments (default: data/{sampleset}/ancestry.tsv).")
    args = parser.parse_args()
    cohort_dir = Path("data") / args.sampleset
    score_dir = cohort_dir / "pgsc" / args.sampleset / "score"

    if args.command == "scores":
        if args.scores_file is None:
            args.scores_file = score_dir / f"{args.sampleset}_pgs.txt.gz"
        if args.output_dir is None:
            args.output_dir = cohort_dir / "source_scores"
    else:
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
        if args.command == "scores":
            count, participants = extract_scores(args.scores_file, args.sampleset, args.source_table,
                                                 args.output_dir, args.analyses)
            print(f"Wrote {count} score panels for {participants:,} participants to {args.output_dir}")
        else:
            covariate_count, ancestry_count = extract_ancestry(
                args.popsimilarity, args.sampleset, args.covariates, args.covariates_out, args.ancestry_out)
            print(f"Wrote {covariate_count:,} covariate rows and {ancestry_count:,} ancestry rows.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
