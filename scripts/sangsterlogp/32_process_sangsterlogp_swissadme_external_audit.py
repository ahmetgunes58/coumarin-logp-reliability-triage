# -*- coding: utf-8 -*-
"""
Process SangsterLogP external coumarin SwissADME output.

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python scripts\32_process_sangsterlogp_swissadme_external_audit.py
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

EXT_DIR = PROJECT_DIR / "data" / "external" / "SangsterLogP"

PROFILE_FILE = EXT_DIR / "SangsterLogP_external_coumarin_profile.csv"
SWISSADME_FILE = EXT_DIR / "SANG_CMR_EXT_001_Swissadme.csv"

OUT_MERGED = EXT_DIR / "SangsterLogP_external_coumarin_SwissADME_merged.csv"
OUT_PERFORMANCE = EXT_DIR / "SangsterLogP_external_coumarin_predictor_performance.csv"
OUT_NGROUP = EXT_DIR / "SangsterLogP_external_coumarin_Ngroup_summary.csv"
OUT_MATRIX = EXT_DIR / "SangsterLogP_external_coumarin_Ncount_logP_matrix.csv"
OUT_ALERTS = EXT_DIR / "SangsterLogP_external_coumarin_alert_enrichment.csv"
OUT_REPORT = EXT_DIR / "SangsterLogP_external_coumarin_audit_report.txt"


def to_numeric_clean(s):
    return pd.to_numeric(s.replace("n/d", np.nan), errors="coerce")


def performance_metrics(delta):
    delta = pd.to_numeric(delta, errors="coerce").dropna()
    if len(delta) == 0:
        return {
            "n": 0,
            "Bias": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Overestimated_percent": np.nan,
            "Severe_abs_error_percent": np.nan,
        }

    return {
        "n": len(delta),
        "Bias": float(delta.mean()),
        "MAE": float(delta.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(delta ** 2))),
        "Overestimated_percent": float((delta < 0).mean() * 100),
        "Severe_abs_error_percent": float((delta.abs() >= 2.0).mean() * 100),
    }


def r2_score(y_exp, y_pred):
    y_exp = pd.to_numeric(y_exp, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    mask = y_exp.notna() & y_pred.notna()
    if mask.sum() < 2:
        return np.nan
    y = y_exp[mask].to_numpy(dtype=float)
    pred = y_pred[mask].to_numpy(dtype=float)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


if not PROFILE_FILE.exists():
    raise FileNotFoundError(f"Profile file not found:\n{PROFILE_FILE}")

if not SWISSADME_FILE.exists():
    raise FileNotFoundError(
        f"SwissADME output file not found:\n{SWISSADME_FILE}\n\n"
        "Copy the downloaded SwissADME CSV to this path first."
    )

profile = pd.read_csv(PROFILE_FILE)
swiss = pd.read_csv(SWISSADME_FILE)

required_profile_cols = [
    "External_ID",
    "Canonical_SMILES",
    "logP_exp",
    "N_count",
    "N_group",
    "logP_bin",
]

missing_profile = [c for c in required_profile_cols if c not in profile.columns]
if missing_profile:
    raise ValueError(f"Missing profile columns: {missing_profile}")

required_swiss_cols = [
    "Molecule",
    "Canonical SMILES",
    "iLOGP",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos-IT Log P",
    "Consensus Log P",
]

missing_swiss = [c for c in required_swiss_cols if c not in swiss.columns]
if missing_swiss:
    raise ValueError(f"Missing SwissADME columns: {missing_swiss}")

# Rename SwissADME columns
swiss_clean = swiss[
    [
        "Molecule",
        "Canonical SMILES",
        "iLOGP",
        "XLOGP3",
        "WLOGP",
        "MLOGP",
        "Silicos-IT Log P",
        "Consensus Log P",
    ]
].copy()

swiss_clean = swiss_clean.rename(
    columns={
        "Molecule": "External_ID",
        "Canonical SMILES": "SwissADME_SMILES",
        "Silicos-IT Log P": "Silicos_IT",
        "Consensus Log P": "Consensus",
    }
)

for col in ["iLOGP", "XLOGP3", "WLOGP", "MLOGP", "Silicos_IT", "Consensus"]:
    swiss_clean[col] = to_numeric_clean(swiss_clean[col].astype(str))

merged = profile.merge(swiss_clean, on="External_ID", how="left")

# Predictor deltas
predictors = ["iLOGP", "XLOGP3", "WLOGP", "MLOGP", "Silicos_IT", "Consensus"]

for p in predictors:
    merged[f"delta_{p}"] = merged["logP_exp"] - merged[p]
    merged[f"abs_delta_{p}"] = merged[f"delta_{p}"].abs()

# Spread4 definition: same as main manuscript, excludes iLOGP and consensus
spread4_cols = ["XLOGP3", "WLOGP", "MLOGP", "Silicos_IT"]
merged["Spread4"] = merged[spread4_cols].max(axis=1) - merged[spread4_cols].min(axis=1)

merged["Consensus_overestimated"] = merged["delta_Consensus"] < 0
merged["Consensus_severe_error"] = merged["abs_delta_Consensus"] >= 2.0
merged["Consensus_severe_overestimation"] = merged["delta_Consensus"] <= -2.0

# Pre-experimental alert rules.
# Since external set does not currently carry full curated structural-class labels,
# these rules use only N_count, predicted consensus logP, and Spread4.
merged["alert_numeric_high_risk"] = (
    (merged["N_count"] >= 1)
    & (merged["Consensus"] > 3.0)
    & (merged["Spread4"] > 2.0)
)

merged["alert_broad_N_or_spread"] = (
    (merged["N_count"] >= 1)
    & (
        (merged["Consensus"] > 3.0)
        | (merged["Spread4"] > 2.0)
    )
)

merged["lower_concern_screen"] = (
    (merged["N_count"] == 0)
    & (merged["Consensus"] <= 3.0)
    & (merged["Spread4"] <= 2.0)
)

# Performance table
perf_rows = []
for p in predictors:
    delta = merged[f"delta_{p}"]
    row = {"Predictor": p}
    row.update(performance_metrics(delta))
    row["R2"] = r2_score(merged["logP_exp"], merged[p])
    row["Missing_predictions"] = int(merged[p].isna().sum())
    perf_rows.append(row)

perf = pd.DataFrame(perf_rows)

# N group summary for consensus
ngroup = (
    merged.groupby("N_group")
    .agg(
        n=("External_ID", "count"),
        mean_delta=("delta_Consensus", "mean"),
        median_delta=("delta_Consensus", "median"),
        mae=("abs_delta_Consensus", "mean"),
        rmse=("delta_Consensus", lambda x: float(np.sqrt(np.mean(np.asarray(x.dropna()) ** 2))) if x.dropna().size else np.nan),
        overestimated_percent=("Consensus_overestimated", lambda x: float(x.mean() * 100)),
        severe_error_n=("Consensus_severe_error", "sum"),
        severe_overestimation_n=("Consensus_severe_overestimation", "sum"),
        mean_spread4=("Spread4", "mean"),
    )
    .reset_index()
)

# N-count × logP-bin matrix for consensus mean bias
matrix = (
    merged.groupby(["N_group", "logP_bin"])
    .agg(
        n=("External_ID", "count"),
        mean_delta=("delta_Consensus", "mean"),
        mae=("abs_delta_Consensus", "mean"),
        severe_error_n=("Consensus_severe_error", "sum"),
        severe_overestimation_n=("Consensus_severe_overestimation", "sum"),
        mean_spread4=("Spread4", "mean"),
    )
    .reset_index()
)

# Alert enrichment summary
def alert_summary(flag_col):
    flagged = merged[merged[flag_col] == True]
    unflagged = merged[merged[flag_col] == False]

    severe_over_total = int(merged["Consensus_severe_overestimation"].sum())
    severe_error_total = int(merged["Consensus_severe_error"].sum())

    return {
        "Alert": flag_col,
        "flagged_n": len(flagged),
        "unflagged_n": len(unflagged),
        "flagged_overestimated_percent": float(flagged["Consensus_overestimated"].mean() * 100) if len(flagged) else np.nan,
        "flagged_severe_error_n": int(flagged["Consensus_severe_error"].sum()) if len(flagged) else 0,
        "flagged_severe_overestimation_n": int(flagged["Consensus_severe_overestimation"].sum()) if len(flagged) else 0,
        "severe_error_recall_percent": float(flagged["Consensus_severe_error"].sum() / severe_error_total * 100) if severe_error_total else np.nan,
        "severe_overestimation_recall_percent": float(flagged["Consensus_severe_overestimation"].sum() / severe_over_total * 100) if severe_over_total else np.nan,
        "mean_delta_flagged": float(flagged["delta_Consensus"].mean()) if len(flagged) else np.nan,
        "mae_flagged": float(flagged["abs_delta_Consensus"].mean()) if len(flagged) else np.nan,
    }

alerts = pd.DataFrame(
    [
        alert_summary("alert_numeric_high_risk"),
        alert_summary("alert_broad_N_or_spread"),
        alert_summary("lower_concern_screen"),
    ]
)

# Save
merged.to_csv(OUT_MERGED, index=False, encoding="utf-8-sig")
perf.to_csv(OUT_PERFORMANCE, index=False, encoding="utf-8-sig")
ngroup.to_csv(OUT_NGROUP, index=False, encoding="utf-8-sig")
matrix.to_csv(OUT_MATRIX, index=False, encoding="utf-8-sig")
alerts.to_csv(OUT_ALERTS, index=False, encoding="utf-8-sig")

# Report
lines = []
lines.append("SangsterLogP external coumarin SwissADME audit report")
lines.append("=" * 78)
lines.append(f"Profile file:   {PROFILE_FILE}")
lines.append(f"SwissADME file: {SWISSADME_FILE}")
lines.append("")
lines.append("Input integrity")
lines.append("-" * 78)
lines.append(f"Profile rows: {len(profile)}")
lines.append(f"SwissADME rows: {len(swiss)}")
lines.append(f"Merged rows: {len(merged)}")
lines.append(f"Missing Consensus predictions: {int(merged['Consensus'].isna().sum())}")
lines.append(f"Missing iLOGP predictions: {int(merged['iLOGP'].isna().sum())}")
lines.append("")
lines.append("Predictor performance")
lines.append("-" * 78)
lines.append(perf.to_string(index=False))
lines.append("")
lines.append("Consensus summary by N group")
lines.append("-" * 78)
lines.append(ngroup.to_string(index=False))
lines.append("")
lines.append("Consensus N-count × logP-bin matrix")
lines.append("-" * 78)
lines.append(matrix.to_string(index=False))
lines.append("")
lines.append("Pre-experimental alert enrichment")
lines.append("-" * 78)
lines.append(alerts.to_string(index=False))
lines.append("")
lines.append("Interpretation notes")
lines.append("-" * 78)
lines.append("This is a non-overlapping external coumarin audit, not prospective validation.")
lines.append("iLOGP was n/d for all uploaded SwissADME rows and should not be used in the external performance comparison.")
lines.append("Spread4 was calculated as max(XLOGP3, WLOGP, MLOGP, Silicos-IT) - min(XLOGP3, WLOGP, MLOGP, Silicos-IT).")
lines.append("Negative delta values indicate overestimation of lipophilicity.")

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\nSangsterLogP external SwissADME audit completed.")
print(f"Merged output:       {OUT_MERGED}")
print(f"Performance table:   {OUT_PERFORMANCE}")
print(f"N-group summary:     {OUT_NGROUP}")
print(f"2D matrix:           {OUT_MATRIX}")
print(f"Alert enrichment:    {OUT_ALERTS}")
print(f"Report:              {OUT_REPORT}")
print("")
print("\n".join(lines))