"""Raw evaluation loading and shared command-line arguments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


phenotypes = {
    "continuous": [
        "LDL_MA", "LOG_HDL_MA", "TC_MA", "LOG_TG_MA", "DBP_MA", "SBP_MA",
        "HEIGHT_MA", "LOG_BMI_MA", "FEV1_FVC_MA", "WHR_SEX", "URT_SEX",
        "LOG_CRTN_SEX", "LOG_TST_SEX",
    ],
    "binary": [
        "ASTHMA_MA", "T2D_MA", "ASTHMA_MT", "T2D_MT", "T1D_MT", "CAD_MT",
        "HTN_MT", "HF_MT", "AF_MT", "STR_MT", "DEM_MT", "GOUT_MT",
    ],
}
methods = [
    "BLocPRS mb1", "BLocPRS mb3", "BLocPRS mb5", "Nonwindowed MultiPRS",
    "Best single-source PRS",
]
metric_tokens = {"continuous": "incremental_r2", "binary": "nagelkerke_r2"}
scenario_labels = {"ukbb": "ukbb_native", "cartagene": "ukbb_to_cartagene"}
method_labels = {"single": "Best single-source PRS", "multiprs": "Nonwindowed MultiPRS",
                 "fixed_composition": "Fixed-composition regional MultiPRS",
                 "regional_single": "Best regional single-source PRS"}
result_keys = ["phenotype", "cohort", "train_group", "group", "panel", "train_size",
               "covariate_set", "metric", "method", "window_mb", "source"]


# Paths refer to packaged outputs, never to archived research runs.
def parse_args(description, supplementary=False, interpretation=False, collection="main", regional=False):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=root / "data")

    if not interpretation:
        parser.add_argument("--results", type=Path, help="Evaluation TSV or directory of evaluation TSVs.")

    output = root / "figures" / "supplementary" if supplementary else root / "figures"
    parser.add_argument("--output-dir", type=Path, default=output)

    if interpretation:
        parser.add_argument("--analysis-dir", type=Path, help="Output directory of the interpretation analyses.")

    if regional:
        parser.add_argument("--gwas-template",
                            help="Raw GWAS path containing {phenotype}, {chromosome}, {group} and optionally {suffix}.")

    args = parser.parse_args()

    if not interpretation:
        args.results = args.results or args.data_root / "evaluation" / collection

    if interpretation:
        args.analysis_dir = args.analysis_dir or args.data_root / "analyses/interpretation"

    if regional:
        args.gwas_template = args.gwas_template or str(args.analysis_dir / (
            "{phenotype}/gwas/restricted/{group}/chr{chromosome}/gwas.{phenotype}.glm.{suffix}"))

    return args


# Attach one bootstrap table to matching point estimates; conflicting runs must remain separate.
def merge_uncertainty_rows(data):
    duplicated = data.duplicated(result_keys, keep=False)

    if not duplicated.any():
        return data

    combined = [data.loc[~duplicated]]
    for key, rows in data.loc[duplicated].groupby(result_keys, dropna=False, sort=False):
        bootstrapped = rows.get("n_boot", pd.Series(0, index=rows.index)).fillna(0).gt(0)

        if len(rows) != 2 or bootstrapped.sum() != 1:
            raise ValueError(f"Ambiguous duplicate evaluations: {key}; provide one point and one bootstrap row.")

        point, uncertainty = rows.loc[~bootstrapped].iloc[0], rows.loc[bootstrapped].iloc[0]

        if uncertainty.get("uncertainty_method") != "paired_bootstrap":
            raise ValueError(f"Unrecognized uncertainty table for {key}.")

        # Only serialization-level rounding is allowed, not a changed prediction or cohort.
        if not np.isclose(point.value, uncertainty.value, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Bootstrap and point estimates disagree: {key}.")

        for column in ("n", "cases", "controls", "status", "reason", "split", "score", "family"):
            if column in rows:
                values = rows[column].fillna("")

                if values.nunique() != 1:
                    raise ValueError(f"Bootstrap and point {column} disagree: {key}.")

        # Preserve failed uncertainty as NE; never substitute a duplicate successful run.
        combined.append(rows.loc[bootstrapped])

    return pd.concat(combined, ignore_index=True)


# Read only metric tables; prediction files in the same directory are not loaded.
def read_results(path):
    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.rglob("*.tsv"))
    tables = []
    for filename in paths:
        header = pd.read_csv(filename, sep="\t", nrows=0).columns

        if {"method", "metric", "value", "n"}.issubset(header):
            tables.append(pd.read_csv(filename, sep="\t"))

    if not tables:
        raise FileNotFoundError(f"No packaged evaluation tables in {path}")

    data = pd.concat(tables, ignore_index=True)

    # Manuscript prediction panels use held-out UKB and frozen-model CaG transfer.
    expected_split = data.cohort.map({"ukbb": "test", "cartagene": "all"})

    if expected_split.isna().any() or not data.split.eq(expected_split).all():
        raise ValueError("Expected held-out UKB (test) or CARTaGENE transfer (all) evaluations.")

    defaults = {"source": "", "window_mb": 0, "panel": "main", "train_group": "all",
                "group": "all", "train_size": "full", "covariate_set": "standard"}
    for column, default in defaults.items():
        if column not in data:
            data[column] = default
        data[column] = data[column].fillna(default)

    data["train_size"] = data.train_size.astype(str).str.replace(r"\.0$", "", regex=True)
    data["window_mb"] = pd.to_numeric(data.window_mb)
    data["scenario"] = data.cohort.map(scenario_labels)
    data["stratum"] = data.group.replace({"all": "All", "female": "Female", "male": "Male"})
    data["family"] = np.where(data.metric.isin(("nagelkerke_r2", "auroc")), "binary", "continuous")

    if "reference" in data:
        data = data.loc[data.reference.fillna("").eq("")].copy()

    return merge_uncertainty_rows(data)


# Scientific selections are explicit and preserve non-estimable values.
def select_results(data, **filters):
    selected = data
    for column, value in filters.items():
        selected = selected.loc[selected[column].isin(value) if isinstance(value, (list, tuple, set))
                                else selected[column].eq(value)]
    return selected.copy()


def one_row(data, **filters):
    selected = select_results(data, **filters)

    if len(selected) != 1:
        raise ValueError(f"Expected one evaluation row for {filters}; found {len(selected)}")

    return selected.iloc[0]


# Best-source comparators use the observed evaluation stratum, with fixed tie breaks.
def method_row(data, method, mb=0, source=None):
    selected = select_results(data, method=method, window_mb=mb)

    if source is not None:
        selected = selected.loc[selected.source.eq(source)]

    if selected.empty or selected.n.nunique() != 1:
        raise ValueError(f"Missing or unmatched {method} evaluations.")

    if method in {"single", "regional_single"} and source is None:

        # Source ID breaks equal-score ties; missing estimates sort last.
        return selected.sort_values(["value", "source"], ascending=[False, True], na_position="last").iloc[0]

    if len(selected) != 1:
        raise ValueError(f"Ambiguous {method} evaluation; specify a source or panel.")

    return selected.iloc[0]


def display_method(method, window_mb=0):
    return f"BLocPRS mb{int(window_mb)}" if method == "blocprs" else method_labels.get(method, method)


def benchmark_rows(data, cohort, phenotype, group="all", train_group="all", metric=None):
    metric = metric or metric_tokens["binary" if phenotype in phenotypes["binary"] else "continuous"]
    frame = select_results(data, cohort=cohort, phenotype=phenotype, group=group,
                           train_group=train_group, train_size="full", panel="main",
                           covariate_set="standard", metric=metric)

    if set(frame.loc[frame.method.eq("single"), "source"]) != set(source_labels(phenotype)):
        raise ValueError(f"Incomplete source panel: {cohort}/{phenotype}/{group}")

    records = []
    for method, mb in (("single", 0), ("multiprs", 0), ("blocprs", 1), ("blocprs", 3), ("blocprs", 5)):
        row = method_row(frame, method, mb).to_dict()
        row["method_token"] = method
        row["source_label"] = row["source"]
        row["method"] = display_method(method, mb)
        records.append(row)

    result = pd.DataFrame(records)

    if result.n.nunique() != 1:
        raise ValueError(f"Evaluation sample counts differ: {cohort}/{phenotype}/{group}")

    return result


# Circle eligibility retains the paper's case-count and absolute-performance rules.
def load_benchmark(path, cohort, non_eur_eval=False, non_eur_train=False):
    data = read_results(path)
    records = []
    for family, names in phenotypes.items():
        for phenotype in names:

            if cohort == "cartagene" and phenotype in {"LOG_TST_SEX", "DEM_MT"}:
                continue

            group = "non-EUR" if non_eur_eval or non_eur_train else "all"
            frame = benchmark_rows(data, cohort, phenotype, group,
                                   "non-EUR" if non_eur_train else "all")

            # Focused benchmarks omit sparse binary transfers and uniformly weak predictions.
            if cohort == "cartagene" and family == "binary" and frame.cases.iloc[0] < 100:
                continue
            if frame.value.max() < .01 or not frame.value.notna().any():
                continue

            records.append(frame.assign(Phenotype=phenotype, Family=family).rename(
                columns={"method": "Method", "value": "Value"}))

    if not records:
        raise ValueError(f"No eligible benchmark estimates for {cohort}")

    return pd.concat(records, ignore_index=True)


def display_phenotype(name):
    label = str(name)
    logged = label.startswith("LOG_")

    if logged:
        label = label[4:]

    for suffix in ("_MA", "_MT", "_SEX"):
        label = label.removesuffix(suffix)
    label = "FEV1/FVC" if label == "FEV1_FVC" else label.replace("_", " ")
    return label + ("\u2020" if logged else "")


# Source names follow the score tables shipped with the repository.
def source_labels(phenotype):
    name = ("multitrait_prs_table.csv" if phenotype.endswith("_MT") else
            "sex_biased_prs_table.csv" if phenotype.endswith("_SEX") else "multi_ancestry_prs_table.csv")
    table = pd.read_csv(Path(__file__).resolve().parents[1] / "tables" / name)
    rows = table.loc[table.AnalysisID.eq(phenotype), ["PGS", "PGS_Name"]].drop_duplicates()

    if rows.PGS.duplicated().any():
        raise ValueError(f"Ambiguous source names: {phenotype}")

    return dict(zip(rows.PGS, rows.PGS_Name))


# Target sources are designated in the score table, never chosen by plot height.
def target_source(phenotype, sources):
    table = pd.read_csv(Path(__file__).resolve().parents[1] / "tables/multitrait_prs_table.csv")
    rows = table.loc[table.AnalysisID.eq(phenotype) & table.Is_Disease_Matched.astype(str).str.lower().eq("true")]

    if len(rows) != 1 or rows.iloc[0].PGS_Name not in sources:
        raise ValueError(f"Missing designated target source: {phenotype}")

    return rows.iloc[0].PGS_Name


def regional_groups(order):
    core = [p for p in order if "_ADJ_" not in p]
    return [(name, title, [p for p in core if p.endswith(suffix)])
            for name, title, suffix in (("multi_ancestry", "Multi-ancestry", "_MA"),
                                       ("sex_specific", "Sex-specific", "_SEX"),
                                       ("related_traits", "Related traits", "_MT"))]


# Contributions use training-score SDs and the centered regional coefficients.
def load_regional_interpretation(data_root, analysis_dir=None):
    from analyses.interpretation.contributions import summarize_contributions

    continuous = ["LDL_MA", "DBP_MA", "HEIGHT_MA", "LOG_BMI_MA", "URT_SEX", "WHR_SEX",
                  "SBP_ADJ_MA", "LOG_CRTN_SEX", "LOG_HDL_MA", "TC_MA", "DBP_ADJ_MA",
                  "FEV1_FVC_MA", "SBP_MA", "TC_ADJ_MA", "LOG_TG_MA", "LDL_ADJ_MA", "LOG_TST_SEX"]
    binary = ["ASTHMA_MA", "CAD_MT", "AF_MT", "ASTHMA_MT", "GOUT_MT", "HF_MT", "HTN_MT",
              "STR_MT", "T1D_MT", "T2D_MA", "T2D_MT", "DEM_MT"]
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else Path(data_root) / "analyses/interpretation"
    paths = sorted(Path(analysis_dir).glob("*/contributions.tsv"))
    context = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    names = set(context.phenotype)
    required = {name for name in continuous + binary if "_ADJ_" not in name}

    if not required.issubset(names) or names - set(continuous + binary):
        raise ValueError("Interpretation summaries require all 25 core phenotypes; medication-adjusted analyses are optional.")

    if not context.window_size.eq("mb1").all():
        raise ValueError("These interpretation figures require 1-Mb windows.")

    order = {name: index for index, name in enumerate(continuous + binary)}
    context = context.sort_values("phenotype", key=lambda names: names.map(order), kind="stable")
    for phenotype, indices in context.groupby("phenotype", sort=False).groups.items():
        context.loc[indices, "expert_label"] = context.loc[indices, "base"].map(
            source_labels(phenotype)).fillna(context.loc[indices, "expert_label"])
    return summarize_contributions(context)


# Paired uncertainty comes from evaluation outputs, not from marginal standard errors.
def read_comparisons(path):
    path = Path(path)
    paths = ([path.with_name(path.stem + "_comparisons.tsv")] if path.is_file()
             else sorted(path.rglob("*.tsv")))
    tables = []
    for filename in paths:
        header = pd.read_csv(filename, sep="\t", nrows=0).columns

        if {"method", "metric", "reference", "difference"}.issubset(header):
            tables.append(pd.read_csv(filename, sep="\t"))

    if not tables:
        raise FileNotFoundError(f"No paired comparison tables in {path}")

    return pd.concat(tables, ignore_index=True)


# Match contrasts to the displayed sources, including optimistic source selection.
def paired_tests(points, path, target="BLocPRS mb1", baselines=None):
    from math import erfc, sqrt

    baselines = baselines or ["Best single-source PRS", "Nonwindowed MultiPRS"]
    contrasts = read_comparisons(path)
    records = []
    keys = ["phenotype", "cohort", "train_group", "group", "panel", "train_size", "covariate_set", "metric"]
    defaults = {"train_group": "all", "group": "all", "panel": "main",
                "train_size": "full", "covariate_set": "standard"}

    for _, cell in points.groupby(keys, dropna=False, sort=False):
        selected = cell.set_index("method")
        target_row = selected.loc[target]
        for baseline in baselines:
            baseline_row = selected.loc[baseline]
            filters = {key: target_row[key] for key in keys}
            match = contrasts.copy()
            for key, value in filters.items():
                values = match[key].fillna(defaults.get(key, "")).astype(str)
                match = match.loc[values.eq(str(value))]

            match = match.loc[match.method.eq(target_row.method_token)]
            if "window_mb" in match:
                match = match.loc[match.window_mb.eq(target_row.window_mb)]

            if "reference_window_mb" in match:
                match = match.loc[match.reference_window_mb.eq(baseline_row.window_mb)]

            reference = str(baseline_row.method_token)
            keep = match.reference.eq(reference)
            if reference == "single":

                # The paired test must use the same source selected for the displayed point.
                source = str(baseline_row.source)
                keep = (match.reference.eq("source:" + source)
                        | (keep & match.get("reference_source", pd.Series("", index=match.index)).eq(source)))
            match = match.loc[keep]

            if len(match) != 1:
                raise ValueError(f"Missing or duplicate paired comparison: {target}/{baseline}, {filters}")

            row = match.iloc[0]
            delta = target_row.value - baseline_row.value

            if row.n != target_row.n or row.n != baseline_row.n:
                raise ValueError("Paired comparison and point estimates have different sample counts.")

            estimable = str(row.get("status", "OK")).lower() in {"ok", "identical_auc_zero_variance"}

            if estimable and not np.isclose(row.difference, delta, rtol=1e-7, atol=1e-10, equal_nan=True):
                raise ValueError("Paired difference does not match the displayed point estimates.")

            se = row.difference_se

            # AUROC uses saved DeLong P values; R2 uses a Wald test of the paired bootstrap difference.
            p = (float(row.p_value) if row.metric == "auroc" and "p_value" in row else
                 erfc(abs(delta / se) / sqrt(2)) if se > 0 else 1.0 if delta == 0 else np.nan)
            records.append(dict(scenario=target_row.scenario, cohort=target_row.cohort,
                                panel=target_row.panel, phenotype=target_row.phenotype,
                                stratum=target_row.stratum, target=target, method=target, baseline=baseline,
                                observed_delta=delta, bootstrap_se_delta=se, p_wald=p,
                                p_wald_two_sided=p, p=p, n=target_row.n))

    return pd.DataFrame(records)


def attach_uncertainty(data, column="bootstrap_se"):
    data = data.copy()

    if column not in data:
        raise ValueError(f"Missing {column}; run the evaluation uncertainty analysis.")

    estimable = data.value.notna()

    # A finite point estimate can still have non-estimable uncertainty.
    if "uncertainty_status" in data:
        estimable &= data.uncertainty_status.str.upper().eq("OK")

    if not np.isfinite(data.loc[estimable, column]).all() or data[column].lt(0).any():
        raise ValueError(f"Missing or invalid {column} for an estimable result.")

    data["err"] = data[column]
    return data


# Sex-specific source labels come from the supplied score IDs.
def load_sex_results(path):
    data = read_results(path)
    records = []
    for cohort in ("ukbb", "cartagene"):
        for phenotype in ("WHR_SEX", "URT_SEX", "LOG_CRTN_SEX", "LOG_TST_SEX"):

            if cohort == "cartagene" and phenotype == "LOG_TST_SEX":
                continue

            trait = phenotype.removeprefix("LOG_").removesuffix("_SEX")
            for group in ("all", "female", "male"):
                frame = select_results(data, cohort=cohort, phenotype=phenotype, group=group,
                                       train_group="all", train_size="full", panel="main",
                                       covariate_set="standard", metric="incremental_r2")
                for method, mb, source, label in (
                    ("single", 0, f"PGS_{trait}_M", "Male PRS"),
                    ("single", 0, f"PGS_{trait}_F", "Female PRS"),
                    ("multiprs", 0, None, "Nonwindowed MultiPRS"),
                    ("blocprs", 1, None, "BLocPRS mb1"),
                    ("blocprs", 3, None, "BLocPRS mb3"),
                    ("blocprs", 5, None, "BLocPRS mb5")):
                    row = method_row(frame, method, mb, source).to_dict()
                    row.update(method_token=method, method=label)
                    records.append(row)

    values = attach_uncertainty(pd.DataFrame(records))
    if values.groupby(["cohort", "phenotype", "group"]).n.nunique().gt(1).any():
        raise ValueError("Sex-stratified methods evaluate different participant counts.")

    return values, paired_tests(values, path, baselines=["Male PRS", "Female PRS", "Nonwindowed MultiPRS"])


# Pool phenotype-by-draw recovery values, retaining the published random-window IQR.
def local_recovery_band(controls, endpoints):
    band = controls.groupby(["Scenario", "RequestedFractionPct"]).LocalGainRecoveryPct.agg(
        q25=lambda values: np.percentile(values, 25), median="median", q75=lambda values: np.percentile(values, 75)).reset_index()
    ends = endpoints.groupby(["Scenario", "RequestedFractionPct"]).LocalGainRecoveryPct.median().reset_index(name="median")

    # At zero and all windows, random selection adds no variability.
    ends["q25"], ends["q75"] = ends["median"], ends["median"]
    return pd.concat([band, ends]).sort_values(["Scenario", "RequestedFractionPct"])


# Read the same fixed four-phenotype recovery comparison from packaged analyses.
def load_packaged_recovery(analysis_dir, names, scenarios, fractions):
    curves, controls, endpoints = [], [], []
    for scenario in scenarios:
        for phenotype in names:
            path = Path(analysis_dir) / phenotype / "ablation" / scenario / "metrics.tsv"
            frame = pd.read_csv(path, sep="\t")

            if (not frame.phenotype.eq(phenotype).all() or not frame.scenario.eq(scenario).all()
                    or not frame.window_size.eq("mb1").all()):
                raise ValueError(f"Ablation path and metadata disagree: {path}")

            frame = frame.rename(columns={"phenotype": "Phenotype", "scenario": "Scenario"})
            frame["RequestedFractionPct"] = 100 * frame.fraction
            frame["LocalGainRecoveryPct"] = 100 * frame.recovery
            curve = frame.loc[frame["mode"].isin(("anchor_only", "full", "retain_top"))].copy()
            random = frame.loc[frame["mode"].eq("retain_random")].copy()
            ends = curve.loc[curve["mode"].isin(("anchor_only", "full"))].copy()

            if (sorted(curve.RequestedFractionPct) != fractions
                    or random.duplicated(["RequestedFractionPct", "replicate"]).any()
                    or random.groupby("RequestedFractionPct").size().to_dict() != dict.fromkeys(fractions[1:-1], 1000)):
                raise ValueError(f"Incomplete ranked or random-window recovery grid: {path}")

            if (not np.isfinite(curve.LocalGainRecoveryPct).all()
                    or not np.isfinite(random.LocalGainRecoveryPct).all()):
                raise ValueError(f"Local-gain recovery is not estimable: {path}")

            curves.append(curve)
            controls.append(random)
            endpoints.append(ends)
    return pd.concat(curves, ignore_index=True), local_recovery_band(
        pd.concat(controls, ignore_index=True), pd.concat(endpoints, ignore_index=True))


# Recovery uses training-ranked windows and a frozen genome-wide component.
def load_local_recovery(data_root, analysis_dir=None):
    analysis_dir = analysis_dir or Path(data_root) / "analyses/interpretation"
    return load_packaged_recovery(analysis_dir,
                                 ["LDL_MA", "ASTHMA_MA", "LOG_CRTN_SEX", "HEIGHT_MA"],
                                 ["ukbb_native", "full_cag_transfer"], [0, 1, 2.5, 5, 10, 25, 50, 100])


# Supplementary benchmark inputs, including medication-adjusted phenotypes.
compact_methods = ["Best single-source PRS", "Nonwindowed MultiPRS",
                   "BLocPRS mb1", "BLocPRS mb3", "BLocPRS mb5"]
compact_phenotypes = {
    "continuous": ["LDL_MA", "LDL_ADJ_MA", "LOG_HDL_MA", "TC_MA", "TC_ADJ_MA",
                   "LOG_TG_MA", "DBP_MA", "DBP_ADJ_MA", "SBP_MA", "SBP_ADJ_MA",
                   "HEIGHT_MA", "LOG_BMI_MA", "FEV1_FVC_MA", "WHR_SEX", "URT_SEX",
                   "LOG_CRTN_SEX", "LOG_TST_SEX"],
    "binary": phenotypes["binary"],
}
compact_scenarios = ("ukbb_native", "ukbb_to_cartagene")
compact_strata = ("All", "EUR", "non-EUR")
compact_key = ["scenario", "phenotype", "stratum", "method"]


# These grids retain all ancestry strata, including non-estimable results.
def load_compact_benchmarks(path, non_eur=False, metric=None):
    data = read_results(path)
    candidates = ({"binary": phenotypes["binary"]} if metric == "auroc" else
                  phenotypes if non_eur else compact_phenotypes)
    records = []
    for cohort in ("ukbb", "cartagene"):
        for family, names in candidates.items():
            for phenotype in names:

                if cohort == "cartagene" and phenotype in {"LOG_TST_SEX", "DEM_MT"}:
                    continue

                for group in (("non-EUR",) if non_eur else ("all", "EUR", "non-EUR")):
                    records.append(benchmark_rows(data, cohort, phenotype, group,
                                                  "non-EUR" if non_eur else "all", metric))
    return pd.concat(records, ignore_index=True), candidates


def attach_compact_bootstrap(data, path, non_eur=False):
    return attach_uncertainty(data), paired_tests(data, path)


# Independent source maxima are optimistic comparators in both evaluation cohorts.
def load_progression(path, cohort):
    data = read_results(path)
    records = []
    for phenotype, family in core_phenotypes:

        if cohort == "cartagene" and phenotype in {"LOG_TST_SEX", "DEM_MT"}:
            continue

        frame = select_results(data, cohort=cohort, phenotype=phenotype, group="all",
                               train_group="all", train_size="full", panel="main",
                               covariate_set="standard", metric=metric_tokens[family])

        # The two source maxima must compare the same complete panel on the same cohort.
        raw = select_results(frame, method="single", window_mb=0)
        regional = select_results(frame, method="regional_single", window_mb=1)
        expected = set(source_labels(phenotype))

        if set(raw.source) != expected or set(regional.source) != expected:
            raise ValueError(f"Incomplete raw/regional source panel: {phenotype}")

        if not np.isfinite(raw.value).all() or not np.isfinite(regional.value).all():
            raise ValueError(f"Non-estimable source candidates: {phenotype}")

        if pd.concat([raw, regional]).n.nunique() != 1:
            raise ValueError(f"Raw/regional sources use different participant counts: {phenotype}")

        expected_split = "test" if cohort == "ukbb" else "all"

        if not frame.split.eq(expected_split).all():
            raise ValueError(f"Unexpected evaluation split: {phenotype}/{cohort}")

        for method, mb in (("single", 0), ("regional_single", 1), ("multiprs", 0),
                           ("fixed_composition", 1), ("blocprs", 1)):
            row = method_row(frame, method, mb).to_dict()
            row["method_token"] = method
            row["method"] = display_method(method, mb)
            records.append(row)

    data = pd.DataFrame(records)

    if data.groupby("phenotype").n.nunique().gt(1).any():
        raise ValueError("Model-progression methods evaluate different participant counts.")

    return attach_uncertainty(data)


# Regional maps use training moments and explicitly supplied GWAS scans.
regional_specs = (
    dict(key="ldl", phenotype="LDL_MA", chromosome=19, title="LDL-C",
         sources=("EUR", "AFR", "EAS", "AMR", "CSA"), ticks=(0, 15, 30, 45, 60),
         loci=((11, 12, "LDLR"), (44, 46, "APOE"))),
    dict(key="asthma", phenotype="ASTHMA_MA", chromosome=6, title="Asthma",
         sources=("EUR", "AFR", "EAS", "CSA"), ticks=(0, 40, 80, 120, 160),
         loci=((26, 34, "MHC"),)),
    dict(key="creatinine", phenotype="LOG_CRTN_SEX", chromosome=17, title="Creatinine\u2020",
         sources=("Female", "Male"), ticks=(0, 20, 40, 60, 80),
         loci=((19, 20, "SLC47A1"), (37, 38, "CDK12"), (59, 60, "BCAS3"))),
    dict(key="height", phenotype="HEIGHT_MA", chromosome=12, title="Height",
         sources=("EUR", "AFR", "EAS", "AMR", "CSA"), ticks=(0, 30, 60, 90, 120),
         loci=((66, 67, "HMGA2"),)),
)


# Read coefficient and input-SD tracks, with GWAS restricted to modeled windows.
def load_regional_panel(data_root, spec, analysis_dir, gwas_template):
    from analyses.interpretation.window_analysis import gwas_chunks

    phenotype = spec["phenotype"]
    path = Path(analysis_dir) / phenotype / "contributions.tsv"
    effects = pd.read_csv(path, sep="\t")

    if not effects.window_size.eq("mb1").all():
        raise ValueError("These regional maps require 1-Mb windows.")

    effects = effects.loc[effects.CHR.eq(spec["chromosome"])].copy()
    effects["expert_label"] = effects.base.map(source_labels(phenotype)).fillna(effects.expert_label).replace(
        {"PGS_CRTN_F": "Female", "PGS_CRTN_M": "Male"})

    # Source/window uniqueness prevents averaging inconsistent coefficients.
    if effects.duplicated(["expert_label", "window"]).any():
        raise ValueError(f"Duplicate source/window coefficients: {phenotype}")
    if set(effects.expert_label) != set(spec["sources"]):
        raise ValueError(f"Unexpected PRS sources: {phenotype}")

    windows = effects[["window", "start_bp", "end_bp"]].drop_duplicates().sort_values("start_bp")
    starts = windows.start_bp.to_numpy(float)
    ends = windows.end_bp.to_numpy(float)

    if windows.window.duplicated().any() or np.any(ends < starts) or np.any(starts[1:] <= ends[:-1]):
        raise ValueError(f"Conflicting or overlapping windows: {phenotype}")

    def matrix(column):

        # Reindex fixes the source and genomic order before conversion to an unlabeled array.
        return (effects.pivot(index="expert_label", columns="window", values=column)
                .reindex(index=spec["sources"], columns=windows.window).to_numpy(float))

    coefficients = [matrix(column) for column in ("c_eff", "gamma", "eta_centered")]
    variance, sd = matrix("model_prs_var"), matrix("model_prs_sd")

    # Effective coefficients combine the anchor and centered local adjustment.
    if not all(np.isfinite(values).all() for values in (*coefficients, variance, sd)):
        raise ValueError(f"Incomplete coefficient/input-scale tracks: {phenotype}")
    if np.any(sd < 0) or not np.allclose(sd ** 2, variance, rtol=1e-7, atol=1e-12):
        raise ValueError(f"Inconsistent training PRS variance and SD: {phenotype}")
    if not np.allclose(coefficients[0], coefficients[1] + coefficients[2], rtol=1e-7, atol=1e-10):
        raise ValueError(f"Effective coefficient identity failed: {phenotype}")

    short_name = phenotype.removesuffix("_MA").removesuffix("_SEX")
    suffix = "logistic.hybrid" if short_name == "ASTHMA" else "linear"
    gwas = {}
    for source in spec["sources"]:
        group = source.lower() if source in {"Female", "Male"} else source
        path = Path(gwas_template.format(phenotype=phenotype, chromosome=spec["chromosome"],
                                        group=group, suffix=suffix))
        counts = dict(n_failed_tests=0, n_missing_p=0, n_zero_p=0, n_valid_variants=0)

        # Use the concentration analysis's successful-test filter, but retain uncapped logP for maps.
        chunks = [pd.DataFrame({"POS": positions, "LOGP": logp})
                  for positions, logp in gwas_chunks(path, spec["chromosome"], counts)]
        points = pd.concat(chunks, ignore_index=True)

        # GWAS positions and stored windows are both one-based inclusive.
        # Find the last window start at or before each SNP, then check its end to exclude gaps.
        indices = np.searchsorted(starts, points.POS.to_numpy(), side="right") - 1
        covered = (indices >= 0) & (points.POS.to_numpy() <= ends[np.maximum(indices, 0)])
        points = points.loc[covered].sort_values("POS")

        if points.empty:
            raise ValueError(f"No GWAS points in modeled windows: {path}")

        gwas[source] = points

    # Only drawing edges convert inclusive coordinates to half-open intervals.
    return dict(spec, coefficients=coefficients, contributions=[values * sd for values in coefficients],
                variance=variance, sd=sd, starts=(starts - 1) / 1e6, ends=ends / 1e6, gwas=gwas)



core_phenotypes = [
    (name, family) for family, names in {
        "continuous": ["LDL_MA", "DBP_MA", "HEIGHT_MA", "LOG_BMI_MA", "URT_SEX", "WHR_SEX",
                       "LOG_CRTN_SEX", "LOG_HDL_MA", "TC_MA", "FEV1_FVC_MA", "SBP_MA",
                       "LOG_TG_MA", "LOG_TST_SEX"],
        "binary": ["ASTHMA_MA", "CAD_MT", "AF_MT", "ASTHMA_MT", "GOUT_MT", "HF_MT",
                   "HTN_MT", "STR_MT", "T1D_MT", "T2D_MA", "T2D_MT", "DEM_MT"],
    }.items() for name in names
]
metric_labels = {"continuous": r"Incremental $R^2$", "binary": r"Nagelkerke $R^2$"}


# Sample-size tables contain both the full fit and the training subsets.
def load_training_sizes(path):
    data = read_results(path)
    data = select_results(data, group="all", train_group="all", covariate_set="standard", panel="main")
    data = data.loc[data.phenotype.isin([name for name, _ in core_phenotypes])]
    data = data.loc[((data.method.eq("blocprs") & data.window_mb.eq(1))
                     | (data.method.eq("multiprs") & data.window_mb.eq(0)))
                    & data.metric.isin(metric_tokens.values())].copy()
    data["model"] = [display_method(method, mb) for method, mb in zip(data.method, data.window_mb)]
    data["size"] = data.train_size.map(lambda size: size if size == "full" else int(size))

    expected_phenotypes = {(cohort, phenotype) for cohort in ("ukbb", "cartagene")
                           for phenotype, _ in core_phenotypes
                           if cohort == "ukbb" or phenotype not in {"LOG_TST_SEX", "DEM_MT"}}

    if set(zip(data.cohort, data.phenotype)) != expected_phenotypes:
        raise ValueError("Training-size figures require all available core phenotype/cohort pairs.")

    if data.groupby(["cohort", "phenotype"]).n.nunique().gt(1).any():
        raise ValueError("Training-size evaluations have different participant counts.")

    expected = {(size, model) for size in (2500, 5000, 12500, 25000, 50000, 100000, "full")
                for model in ("Nonwindowed MultiPRS", "BLocPRS mb1")}
    for key, cell in data.groupby(["cohort", "phenotype"]):

        if set(zip(cell["size"], cell.model)) != expected:
            raise ValueError(f"Incomplete training-size series: {key}")

    return data


def phenotype_label(name):
    label = display_phenotype(name)
    label = {"HEIGHT": "Height", "LDL": "LDL-C", "HDL\u2020": "HDL-C\u2020",
             "ASTHMA": "Asthma"}.get(label, label)
    if name in {"ASTHMA_MA", "T2D_MA"}:
        label += " (MA)"
    elif name in {"ASTHMA_MT", "T2D_MT"}:
        label += " (RT)"
    return label
