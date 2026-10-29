# Adapted from MoEPRS: https://github.com/shz9/moe-prs-paper
# Modified for BLocPRS.

"""Download selected PGS Catalog harmonized weights without changing their contents."""

import argparse
import csv
import json
from pathlib import Path
import re
import urllib.request


def read_pgs_ids(pgs_ids=None, pgs_table=None):
    if pgs_table is not None:
        with Path(pgs_table).open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            pgs_ids = [row["PGSCatalog_ID"] for row in rows if row["PGSCatalog_ID"]]

    pgs_ids = list(dict.fromkeys(str(value).strip() for value in pgs_ids if str(value).strip()))

    if not pgs_ids or any(re.fullmatch(r"PGS[0-9]{6}", value) is None for value in pgs_ids):
        raise ValueError("Supply Catalog IDs such as PGS000001; custom weights are not downloaded here.")

    return pgs_ids


# Download to a temporary filename so interrupted transfers are not treated as complete.
def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_weights(pgs_ids, builds, output_dir):
    builds = list(dict.fromkeys(builds))
    output_dir = Path(output_dir)
    count = 0
    for pgs_id in dict.fromkeys(pgs_ids):
        pending = [build for build in builds if not (output_dir / build / f"{pgs_id}.txt.gz").is_file()]
        if not pending:
            continue
        with urllib.request.urlopen(f"https://www.pgscatalog.org/rest/score/{pgs_id}", timeout=60) as response:
            metadata = json.load(response)

        for build in pending:
            url = metadata["ftp_harmonized_scoring_files"][build]["positions"]
            download_file(url, output_dir / build / f"{pgs_id}.txt.gz")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--pgs-ids", nargs="+", help="Catalog IDs, for example PGS000001 PGS000002.")
    sources.add_argument("--pgs-table", type=Path, help="CSV with a PGSCatalog_ID column; blank entries are skipped.")
    parser.add_argument("--builds", nargs="+", choices=("GRCh37", "GRCh38"), default=["GRCh37"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/weights"),
                        help="Weight directory (default: data/weights).")
    args = parser.parse_args()

    try:
        pgs_ids = read_pgs_ids(args.pgs_ids, args.pgs_table)
        count = download_weights(pgs_ids, args.builds, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Downloaded {count} new scoring files to {args.output_dir}; existing files were skipped.")


if __name__ == "__main__":
    main()
