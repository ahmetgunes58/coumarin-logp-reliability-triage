# -*- coding: utf-8 -*-
"""
Internal prospective-alert audit for Figure 8 reliability-triage workflow.

Purpose:
- Test whether pre-experimental alert rules enrich severe overestimation cases.
- Use only information available before experimental logP where possible:
  N_count, SwissADME consensus logP, Spread4, and structural/chemical flags.
- Do NOT present this as an externally validated classifier.
- Output SI-ready tables for Table S8c and supporting datasets.

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python run_prospective_alert_audit.py
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DATA = ROOT / "data" / "processed"
DATA.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "Dataset_S1_benchmark_dataset.csv"

OUT_COMPOUND = DATA / "Dataset_S30_prospective_alert_compound_flags.csv"
OUT_SUMMARY = DATA / "Dataset_S31_prospective_alert_performance_summary.csv"
OUT_REPORT = DATA / "Dataset_S30_S31_prospective_alert_audit_report.txt"

if not INPUT.exists():
    raise FileNotFoundError(f"Dataset bulunamadı: {INPUT}")


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_csv(INPUT)

required = [
    "Compound_ID",
    "SMILES",
    "N_count",
    "logP_exp",
    "Consensus",
    "delta_Consensus",
    "FM",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Eksik gerekli kolonlar: {missing}\nMevcut kolonlar: {list(df.columns)}")

for c in ["N_count", "logP_exp", "Consensus", "delta_Consensus"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

fragment_cols = ["XLOGP3", "WLOGP", "MLOGP", "Silicos_IT"]
missing_frag = [c for c in fragment_cols if c not in df.columns]
if missing_frag:
    raise KeyError(f"Spread4 hesaplamak için eksik kolonlar: {missing_frag}")

for c in fragment_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Spread4"] = df[fragment_cols].max(axis=1) - df[fragment_cols].min(axis=1)

# Main outcomes
df["overestimated"] = df["delta_Consensus"] < 0
df["severe_error_abs_ge_2"] = df["delta_Consensus"].abs() >= 2.0
df["severe_overestimation"] = df["delta_Consensus"] <= -2.0
df["severe_underestimation"] = df["delta_Consensus"] >= 2.0

# Useful optional descriptors
if "Coumarin_Type" not in df.columns:
    df["Coumarin_Type"] = ""
if "Structural_Class" not in df.columns:
    df["Structural_Class"] = ""

df["Coumarin_Type"] = df["Coumarin_Type"].astype(str)
df["Structural_Class"] = df["Structural_Class"].astype(str)


# ============================================================
# 3. Prospective alert rules
# ============================================================
# These are internal operational alert rules, not universal thresholds.

# Core numerical thresholds explored internally
SPREAD_HIGH = 2.0
CONS_HIGH = 3.0

# Structural proxies available from curated annotations / structure inspection.
# Conservative text-matching helper.
def contains_any(series, terms):
    s = series.astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for term in terms:
        mask = mask | s.str.contains(term.lower(), regex=False, na=False)
    return mask

struct_text = (df["Coumarin_Type"].fillna("") + " " + df["Structural_Class"].fillna("")).astype(str)

df["alert_high_spread4"] = df["Spread4"] > SPREAD_HIGH
df["alert_high_consensus"] = df["Consensus"] > CONS_HIGH
df["alert_n_bearing"] = df["N_count"] >= 1
df["alert_n_1_3"] = df["N_count"].between(1, 3, inclusive="both")
df["alert_n_ge_4"] = df["N_count"] >= 4
df["alert_n_free"] = df["N_count"] == 0

# Structural alerts based on annotations.
df["alert_conjugated_or_DA"] = contains_any(
    struct_text,
    [
        "conjugated",
        "donor",
        "acceptor",
        "phosphonate",
        "7-amino",
    ],
)

df["alert_compact_contained"] = contains_any(
    struct_text,
    [
        "oxadiazole",
        "oxadiazoline",
        "furocoumarin",
        "amidoxime",
    ],
)

df["alert_pi_extended_dimer"] = contains_any(
    struct_text,
    [
        "dimer",
        "dimeric",
    ],
)

# Main Figure 8-like prospective alerts
df["A1_high_spread4"] = df["alert_high_spread4"]
df["A2_high_consensus"] = df["alert_high_consensus"]
df["A3_high_spread4_and_high_consensus"] = df["alert_high_spread4"] & df["alert_high_consensus"]
df["A4_n_bearing_high_spread4_high_consensus"] = (
    df["alert_n_bearing"] & df["alert_high_spread4"] & df["alert_high_consensus"]
)
df["A5_n_bearing_conjugated_high_spread4_or_high_consensus"] = (
    df["alert_n_bearing"] &
    df["alert_conjugated_or_DA"] &
    (df["alert_high_spread4"] | df["alert_high_consensus"])
)
df["A6_n_1_3_conjugated_high_spread4_or_high_consensus"] = (
    df["alert_n_1_3"] &
    df["alert_conjugated_or_DA"] &
    (df["alert_high_spread4"] | df["alert_high_consensus"])
)
df["A7_n_ge_4_mixed_highN_provisional"] = df["alert_n_ge_4"]
df["A8_n_free_pi_extended_boundary"] = df["alert_n_free"] & df["alert_pi_extended_dimer"]

# Lower-concern screen, intentionally conservative
df["A9_lower_concern_screen"] = (
    df["alert_compact_contained"] &
    (~df["alert_high_spread4"]) &
    (~df["alert_high_consensus"])
)


alert_definitions = [
    (
        "A1_high_spread4",
        f"Spread4 > {SPREAD_HIGH}",
        "Predictor-disagreement alert",
    ),
    (
        "A2_high_consensus",
        f"Consensus logP > {CONS_HIGH}",
        "High predicted-lipophilicity alert",
    ),
    (
        "A3_high_spread4_and_high_consensus",
        f"Spread4 > {SPREAD_HIGH} and consensus logP > {CONS_HIGH}",
        "Combined numerical overestimation-risk alert",
    ),
    (
        "A4_n_bearing_high_spread4_high_consensus",
        f"N_count ≥ 1, Spread4 > {SPREAD_HIGH}, and consensus logP > {CONS_HIGH}",
        "N-bearing high-disagreement / high-predicted-logP alert",
    ),
    (
        "A5_n_bearing_conjugated_high_spread4_or_high_consensus",
        f"N-bearing conjugated/donor-acceptor motif with Spread4 > {SPREAD_HIGH} or consensus logP > {CONS_HIGH}",
        "Structure-informed potential overestimation-risk alert",
    ),
    (
        "A6_n_1_3_conjugated_high_spread4_or_high_consensus",
        f"N = 1–3 conjugated/donor-acceptor motif with Spread4 > {SPREAD_HIGH} or consensus logP > {CONS_HIGH}",
        "FM1/FM2-like prospective overestimation-risk alert",
    ),
    (
        "A7_n_ge_4_mixed_highN_provisional",
        "N_count ≥ 4",
        "Mixed high-N provisional alert",
    ),
    (
        "A8_n_free_pi_extended_boundary",
        "N = 0 and π-extended dimeric motif",
        "N-free π-extended boundary alert",
    ),
    (
        "A9_lower_concern_screen",
        f"Compact electronically contained motif, Spread4 ≤ {SPREAD_HIGH}, and consensus logP ≤ {CONS_HIGH}",
        "Lower-concern screen",
    ),
]


# ============================================================
# 4. Metric functions
# ============================================================

total_n = len(df)
total_severe_over = int(df["severe_overestimation"].sum())
total_severe_abs = int(df["severe_error_abs_ge_2"].sum())


def safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b


def summarize_alert(flag_col, rule, interpretation):
    flagged = df[df[flag_col]].copy()
    not_flagged = df[~df[flag_col]].copy()

    flagged_n = len(flagged)
    severe_over_n = int(flagged["severe_overestimation"].sum())
    severe_abs_n = int(flagged["severe_error_abs_ge_2"].sum())
    overestimated_n = int(flagged["overestimated"].sum())

    precision_severe_over = safe_div(severe_over_n, flagged_n)
    recall_severe_over = safe_div(severe_over_n, total_severe_over)

    precision_severe_abs = safe_div(severe_abs_n, flagged_n)
    recall_severe_abs = safe_div(severe_abs_n, total_severe_abs)

    return {
        "Alert_ID": flag_col,
        "Prospective_alert_rule": rule,
        "Interpretation": interpretation,
        "Flagged_n": flagged_n,
        "Flagged_percent": 100 * safe_div(flagged_n, total_n),
        "Mean_bias_flagged": flagged["delta_Consensus"].mean() if flagged_n else np.nan,
        "Median_bias_flagged": flagged["delta_Consensus"].median() if flagged_n else np.nan,
        "MAE_flagged": flagged["delta_Consensus"].abs().mean() if flagged_n else np.nan,
        "RMSE_flagged": np.sqrt(np.mean(flagged["delta_Consensus"] ** 2)) if flagged_n else np.nan,
        "Overestimated_n_flagged": overestimated_n,
        "Overestimated_percent_flagged": 100 * safe_div(overestimated_n, flagged_n),
        "Severe_overestimation_n_flagged": severe_over_n,
        "Severe_overestimation_precision_percent": 100 * precision_severe_over,
        "Severe_overestimation_recall_percent": 100 * recall_severe_over,
        "Severe_abs_error_n_flagged": severe_abs_n,
        "Severe_abs_error_precision_percent": 100 * precision_severe_abs,
        "Severe_abs_error_recall_percent": 100 * recall_severe_abs,
        "Mean_bias_unflagged": not_flagged["delta_Consensus"].mean() if len(not_flagged) else np.nan,
        "Overestimated_percent_unflagged": 100 * safe_div(int(not_flagged["overestimated"].sum()), len(not_flagged)),
    }


summary_rows = [
    summarize_alert(flag, rule, interpretation)
    for flag, rule, interpretation in alert_definitions
]

summary = pd.DataFrame(summary_rows)

# Round for readability
round_cols = [
    "Flagged_percent",
    "Mean_bias_flagged",
    "Median_bias_flagged",
    "MAE_flagged",
    "RMSE_flagged",
    "Overestimated_percent_flagged",
    "Severe_overestimation_precision_percent",
    "Severe_overestimation_recall_percent",
    "Severe_abs_error_precision_percent",
    "Severe_abs_error_recall_percent",
    "Mean_bias_unflagged",
    "Overestimated_percent_unflagged",
]
for c in round_cols:
    summary[c] = pd.to_numeric(summary[c], errors="coerce").round(2)


# Compound-level output columns
compound_cols = [
    "Compound_ID",
    "SMILES",
    "FM",
    "N_count",
    "logP_exp",
    "Consensus",
    "delta_Consensus",
    "Spread4",
    "Coumarin_Type",
    "Structural_Class",
    "overestimated",
    "severe_overestimation",
    "severe_error_abs_ge_2",
] + [flag for flag, _, _ in alert_definitions]

compound_cols = [c for c in compound_cols if c in df.columns]

df[compound_cols].to_csv(OUT_COMPOUND, index=False, encoding="utf-8-sig")
summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")


# ============================================================
# 5. Report
# ============================================================

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("Internal prospective-alert audit for Figure 8 reliability-triage workflow\n")
    f.write("=" * 86 + "\n\n")
    f.write(f"Input dataset: {INPUT}\n")
    f.write(f"Dataset n: {total_n}\n")
    f.write("Error convention: delta = experimental logP - predicted consensus logP\n")
    f.write("Negative delta indicates overestimation.\n\n")
    f.write(f"Severe overestimation definition: delta_Consensus <= -2.0\n")
    f.write(f"Severe overestimation count: {total_severe_over}\n")
    f.write(f"Severe absolute error definition: |delta_Consensus| >= 2.0\n")
    f.write(f"Severe absolute error count: {total_severe_abs}\n\n")
    f.write("Important interpretation note:\n")
    f.write(
        "These are internal alert checks, not externally validated classifier results. "
        "The thresholds are operational triage thresholds explored within the present coumarin dataset.\n\n"
    )
    f.write("Prospective-alert performance summary:\n")
    f.write(summary.to_string(index=False))
    f.write("\n\n")
    f.write("Outputs:\n")
    f.write(str(OUT_COMPOUND) + "\n")
    f.write(str(OUT_SUMMARY) + "\n")
    f.write(str(OUT_REPORT) + "\n")


# ============================================================
# 6. Console output
# ============================================================

print("\nInternal prospective-alert audit tamamlandı.")
print(f"Input    : {INPUT}")
print(f"Compound : {OUT_COMPOUND}")
print(f"Summary  : {OUT_SUMMARY}")
print(f"Report   : {OUT_REPORT}")
print("\nDataset-level severe outcomes:")
print(f"  severe overestimation n = {total_severe_over}")
print(f"  severe abs error n      = {total_severe_abs}")
print("\nProspective-alert summary:")
show_cols = [
    "Alert_ID",
    "Flagged_n",
    "Mean_bias_flagged",
    "Overestimated_percent_flagged",
    "Severe_overestimation_n_flagged",
    "Severe_overestimation_precision_percent",
    "Severe_overestimation_recall_percent",
    "Severe_abs_error_n_flagged",
    "Severe_abs_error_precision_percent",
    "Severe_abs_error_recall_percent",
]
print(summary[show_cols].to_string(index=False))