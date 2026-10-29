# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Prepare UK Biobank sample IDs and native-PC covariates from tabular data."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def validate_ids(ids, label):

    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{label} contains missing participant IDs.")

    if ids.str.contains(r"\s").any():
        raise ValueError(f"{label} contains whitespace in participant IDs; correct the input before matching.")

    if ids.duplicated().any():
        raise ValueError(f"{label} contains duplicate participant IDs.")


# Read a headerless withdrawal list; an empty file means no withdrawals.
def read_withdrawals(path):
    try:
        table = pd.read_csv(path, header=None, dtype=str, skip_blank_lines=False)
    except pd.errors.EmptyDataError:
        return set()

    if table.shape[1] != 1:
        raise ValueError("The withdrawal file must contain one IID column without a header.")

    validate_ids(table.iloc[:, 0], "Withdrawal file")
    return set(table.iloc[:, 0])


def prepare_samples(tabular, withdrawals, keep_out, covariates_out, num_pcs=20):

    if num_pcs < 1:
        raise ValueError("num_pcs must be at least 1.")

    tabular, withdrawals, keep_out, covariates_out = [
        Path(path).expanduser() for path in (tabular, withdrawals, keep_out, covariates_out)
    ]

    if len({path.resolve() for path in (tabular, withdrawals, keep_out, covariates_out)}) != 4:
        raise ValueError("Input and output paths must be distinct.")

    for path in (keep_out, covariates_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    withdrawn = read_withdrawals(withdrawals)
    pc_columns = [f"PC{i}" for i in range(1, num_pcs + 1)]
    fields = ["eid", "22001-0.0", "21003-0.0", "22019-0.0", "22021-0.0"]
    fields += [f"22009-0.{i}" for i in range(1, num_pcs + 1)]
    names = ["IID", "Sex", "Age", "Aneuploidy", "Relatedness"] + pc_columns
    dtypes = {field: float for field in fields[1:]}
    dtypes["eid"] = str
    seen_ids = set()
    counts = {"read": 0, "kept": 0, "covariates": 0}

    with TemporaryDirectory(dir=keep_out.parent, prefix=".keep_") as keep_dir, \
         TemporaryDirectory(dir=covariates_out.parent, prefix=".covariates_") as covariates_dir:
        keep_temporary = Path(keep_dir) / "keep.tsv"
        covariates_temporary = Path(covariates_dir) / "covariates.tsv"
        with keep_temporary.open("w", encoding="utf-8") as keep_handle, \
             covariates_temporary.open("w", encoding="utf-8") as covariates_handle:
            chunks = pd.read_csv(tabular, usecols=fields, dtype=dtypes, chunksize=100000, skip_blank_lines=False)
            for chunk in chunks:
                chunk = chunk[fields]
                chunk.columns = names
                validate_ids(chunk["IID"], "UK Biobank table")
                ids = set(chunk["IID"])

                if not seen_ids.isdisjoint(ids):
                    raise ValueError("UK Biobank table contains duplicate participant IDs across row blocks.")

                seen_ids.update(ids)
                counts["read"] += len(chunk)

                # Missing aneuploidy is retained; relatedness must be 0 or 1.
                eligible = ~chunk["IID"].isin(withdrawn)
                eligible &= chunk["Relatedness"].isin([0, 1]) & chunk["Aneuploidy"].ne(1)
                samples = chunk.loc[eligible].copy()
                samples.insert(0, "FID", samples["IID"])

                # Genotype eligibility does not require complete covariates.
                samples[["FID", "IID"]].to_csv(keep_handle, sep="\t", header=False, index=False)
                counts["kept"] += len(samples)

                # These are native UKB PCs, not the projected reference PCs.
                covariates = samples[["FID", "IID", "Sex"] + pc_columns + ["Age"]].dropna().copy()
                covariates[pc_columns] = covariates[pc_columns].round(5)
                covariates.to_csv(covariates_handle, sep="\t", header=False, index=False)
                counts["covariates"] += len(covariates)

        # Replace outputs only after all input blocks pass QC.
        keep_temporary.replace(keep_out)
        covariates_temporary.replace(covariates_out)

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular", required=True, help="UKB tabular CSV with eid and the required QC/PC fields.")
    parser.add_argument("--withdrawals", required=True, help="Headerless CSV with one withdrawn IID per row.")
    parser.add_argument("--keep-out", required=True, help="Headerless FID/IID keep file.")
    parser.add_argument("--covariates-out", required=True, help="Headerless FID/IID/Sex/PC1..PCN/Age file.")
    parser.add_argument("--num-pcs", type=int, default=20, help="Number of native UKB PCs to extract (default: 20).")
    args = parser.parse_args()
    try:
        counts = prepare_samples(args.tabular, args.withdrawals, args.keep_out, args.covariates_out, args.num_pcs)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Samples read: {counts['read']}; kept: {counts['kept']}; complete covariates: {counts['covariates']}.")


if __name__ == "__main__":
    main()
