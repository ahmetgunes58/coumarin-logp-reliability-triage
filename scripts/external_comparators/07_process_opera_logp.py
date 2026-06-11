# -*- coding: utf-8 -*-
"""
07_process_opera_logp.py

Purpose
-------
Process OPERA 2.9 logP raw output and calculate external comparator
performance metrics.

Inputs
------
data/processed/Dataset_S42_external_predictor_input.csv
data/processed/OPERA_logP_raw_output.csv

Outputs
-------
data/processed/Dataset_S42d_OPERA_logP_values.csv
data/processed/Dataset_S42d_OPERA_logP_report.txt

Important
---------
OPERA is used as an external platform-independent comparator.
It is not used to train, recalibrate, or replace any SwissADME-associated predictor.
"""

from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]

BASE_INPUT = PROJECT_DIR / "data" / "processed" / "Dataset_S42_external_predictor_input.csv"
OPERA_RAW = PROJECT_DIR / "data" / "processed" / "OPERA_logP_raw_output.csv"

OUT_VALUES = PROJECT_DIR / "data" / "processed" / "Dataset_S42d_OPERA_logP_values.csv"
OUT_REPORT = PROJECT_DIR / "data" / "processed" / "Dataset_S42d_OPERA_logP_report.txt"


def calculate_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def calculate_metrics(df: pd.DataFrame, pred_col: str, mask: pd.Series | None = None) -> dict:
    if mask is None:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask].copy()
    valid = sub[pred_col].notna() & sub["logP_exp"].notna()

    if valid.sum() == 0:
        return {
            "n": 0,
            "Bias": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Overestimated_percent": np.nan,
            "Severe_abs_error_percent": np.nan,
        }

    y_true = sub.loc[valid, "logP_exp"].astype(float)
    y_pred = sub.loc[valid, pred_col].astype(float)
    delta = y_true - y_pred

    return {
        "n": int(valid.sum()),
        "Bias": float(delta.mean()),
        "MAE": float(delta.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(delta ** 2))),
        "R2": calculate_r2(y_true, y_pred),
        "Overestimated_percent": float((delta < 0).mean() * 100),
        "Severe_abs_error_percent": float((delta.abs() >= 2.0).mean() * 100),
    }


def main() -> None:
    if not BASE_INPUT.exists():
        raise FileNotFoundError(f"Base input not found:\n{BASE_INPUT}")

    if not OPERA_RAW.exists():
        raise FileNotFoundError(f"OPERA raw output not found:\n{OPERA_RAW}")

    base = pd.read_csv(BASE_INPUT, encoding="utf-8-sig")
    opera = pd.read_csv(OPERA_RAW, encoding="utf-8-sig")

    required_base = ["Compound_ID", "SMILES", "logP_exp", "N_count", "FM"]
    required_opera = [
        "MoleculeID",
        "LogP_pred",
        "LogP_predRange",
        "AD_LogP",
        "AD_index_LogP",
        "Conf_index_LogP",
    ]

    missing_base = [c for c in required_base if c not in base.columns]
    missing_opera = [c for c in required_opera if c not in opera.columns]

    if missing_base:
        raise ValueError(f"Missing columns in base input: {missing_base}")

    if missing_opera:
        raise ValueError(f"Missing columns in OPERA raw output: {missing_opera}")

    if base["Compound_ID"].duplicated().any():
        raise ValueError("Duplicate Compound_ID values detected in base input.")

    if opera["MoleculeID"].duplicated().any():
        duplicates = opera.loc[opera["MoleculeID"].duplicated(), "MoleculeID"].tolist()
        raise ValueError(f"Duplicate MoleculeID values detected in OPERA output: {duplicates[:10]}")

    opera_small = opera[required_opera].copy()
    opera_small = opera_small.rename(
        columns={
            "MoleculeID": "Compound_ID",
            "LogP_pred": "OPERA_logP",
            "LogP_predRange": "OPERA_logP_predRange",
            "AD_LogP": "OPERA_AD_LogP",
            "AD_index_LogP": "OPERA_AD_index_LogP",
            "Conf_index_LogP": "OPERA_Conf_index_LogP",
        }
    )

    numeric_cols = [
        "OPERA_logP",
        "OPERA_AD_LogP",
        "OPERA_AD_index_LogP",
        "OPERA_Conf_index_LogP",
    ]

    for col in numeric_cols:
        opera_small[col] = pd.to_numeric(opera_small[col], errors="coerce")

    merged = base.merge(
        opera_small,
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )

    missing_opera_values = merged["OPERA_logP"].isna().sum()
    if missing_opera_values > 0:
        missing_ids = merged.loc[merged["OPERA_logP"].isna(), "Compound_ID"].tolist()
        raise ValueError(
            f"Missing OPERA logP values for {missing_opera_values} compounds.\n"
            f"First missing IDs: {missing_ids[:20]}"
        )

    merged["logP_exp"] = pd.to_numeric(merged["logP_exp"], errors="raise")
    merged["N_count"] = pd.to_numeric(merged["N_count"], errors="raise")

    merged["delta_OPERA_logP"] = merged["logP_exp"] - merged["OPERA_logP"]
    merged["abs_delta_OPERA_logP"] = merged["delta_OPERA_logP"].abs()

    out_cols = [
        "Compound_ID",
        "SMILES",
        "logP_exp",
        "N_count",
        "FM",
        "OPERA_logP",
        "OPERA_logP_predRange",
        "OPERA_AD_LogP",
        "OPERA_AD_index_LogP",
        "OPERA_Conf_index_LogP",
        "delta_OPERA_logP",
        "abs_delta_OPERA_logP",
    ]

    out = merged[out_cols].copy()
    OUT_VALUES.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_VALUES, index=False, encoding="utf-8")

    full_metrics = calculate_metrics(out, "OPERA_logP")

    ad_mask = out["OPERA_AD_LogP"].eq(1)
    non_ad_mask = out["OPERA_AD_LogP"].eq(0)
    polar_mask = out["logP_exp"] <= 1.0
    high_risk_mask = out["N_count"].between(1, 3) & (out["logP_exp"] < 1.5)
    fm1_mask = out["FM"].astype(str).eq("FM1")
    fm2_mask = out["FM"].astype(str).eq("FM2")

    ad_metrics = calculate_metrics(out, "OPERA_logP", ad_mask)
    non_ad_metrics = calculate_metrics(out, "OPERA_logP", non_ad_mask)
    polar_metrics = calculate_metrics(out, "OPERA_logP", polar_mask)
    high_risk_metrics = calculate_metrics(out, "OPERA_logP", high_risk_mask)
    fm1_metrics = calculate_metrics(out, "OPERA_logP", fm1_mask)
    fm2_metrics = calculate_metrics(out, "OPERA_logP", fm2_mask)

    report = []
    report.append("Dataset_S42d OPERA 2.9 logP processing report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Base input file: {BASE_INPUT}")
    report.append(f"OPERA raw output: {OPERA_RAW}")
    report.append(f"Output values: {OUT_VALUES}")
    report.append(f"Output report: {OUT_REPORT}")
    report.append("")
    report.append("1. Processing status")
    report.append("-" * 70)
    report.append(f"Base compounds: {len(base)}")
    report.append(f"OPERA raw output rows: {len(opera)}")
    report.append(f"Valid OPERA logP values: {int(out['OPERA_logP'].notna().sum())}")
    report.append(f"Missing OPERA logP values: {int(missing_opera_values)}")
    report.append("")
    report.append("2. OPERA applicability-domain summary")
    report.append("-" * 70)
    report.append(f"AD_LogP = 1 compounds: {int(ad_mask.sum())}")
    report.append(f"AD_LogP = 0 compounds: {int(non_ad_mask.sum())}")
    report.append(f"Mean OPERA AD index: {out['OPERA_AD_index_LogP'].mean():.3f}")
    report.append(f"Mean OPERA confidence index: {out['OPERA_Conf_index_LogP'].mean():.3f}")
    report.append("")
    report.append("3. OPERA full-dataset performance summary")
    report.append("-" * 70)
    for key, value in full_metrics.items():
        if key == "n":
            report.append(f"{key}: {value}")
        else:
            report.append(f"{key}: {value:.3f}")
    report.append("")
    report.append("4. OPERA subset performance summary")
    report.append("-" * 70)

    subset_blocks = {
        "AD_LogP = 1": ad_metrics,
        "AD_LogP = 0": non_ad_metrics,
        "Polar logP <= 1.0": polar_metrics,
        "N = 1-3 and logP < 1.5": high_risk_metrics,
        "FM1": fm1_metrics,
        "FM2": fm2_metrics,
    }

    for label, metrics in subset_blocks.items():
        report.append(label)
        for key, value in metrics.items():
            if key == "n":
                report.append(f"  {key}: {value}")
            else:
                report.append(f"  {key}: {value:.3f}")
        report.append("")

    report.append("5. Interpretation note")
    report.append("-" * 70)
    report.append(
        "OPERA logP is included as an external platform-independent comparator. "
        "The OPERA applicability-domain fields are retained in the output table "
        "to support transparent interpretation. OPERA is not used to train, "
        "recalibrate, or replace any SwissADME-associated predictor."
    )

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("SUCCESS: OPERA logP processing completed.")


if __name__ == "__main__":
    main()