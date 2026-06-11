# -*- coding: utf-8 -*-
"""
02_process_datawarrior_clogp.py

Purpose
-------
Validate and standardise DataWarrior cLogP export for the external
platform-independent logP comparator audit.

Input
-----
data/processed/DataWarrior_cLogP_export.csv

Output
------
data/processed/Dataset_S42b_DataWarrior_cLogP_values.csv
data/processed/Dataset_S42b_DataWarrior_cLogP_report.txt

Important
---------
This script does NOT modify Dataset_S1_benchmark_dataset.csv.
It validates the DataWarrior export against:
    data/processed/Dataset_S42_external_predictor_input.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]

BASE_INPUT = PROJECT_DIR / "data" / "processed" / "Dataset_S42_external_predictor_input.csv"
DATAWARRIOR_EXPORT = PROJECT_DIR / "data" / "processed" / "DataWarrior_cLogP_export.csv"

OUT_VALUES = PROJECT_DIR / "data" / "processed" / "Dataset_S42b_DataWarrior_cLogP_values.csv"
OUT_REPORT = PROJECT_DIR / "data" / "processed" / "Dataset_S42b_DataWarrior_cLogP_report.txt"


POSSIBLE_CLOGP_COLUMNS = [
    "cLogP",
    "clogP",
    "CLogP",
    "cLogP (DataWarrior)",
    "DataWarrior_cLogP",
    "Calculated LogP",
    "LogP",
    "logP",
]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Try common CSV/TSV separators and return the best-looking DataFrame.
    """
    attempts = []
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            attempts.append((sep, df))
        except Exception:
            continue

    if not attempts:
        raise ValueError(f"Could not read file: {path}")

    # choose the parse with the largest number of columns
    sep, df = max(attempts, key=lambda x: len(x[1].columns))
    return df


def find_clogp_column(df: pd.DataFrame) -> str:
    for col in POSSIBLE_CLOGP_COLUMNS:
        if col in df.columns:
            return col

    # Fallback: find any column containing both "log" and "p"
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if "log" in lower and "p" in lower:
            candidates.append(col)

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Could not identify DataWarrior cLogP column.\n"
        f"Available columns are:\n{list(df.columns)}\n\n"
        "Please rename the DataWarrior cLogP column to exactly: cLogP"
    )


def calculate_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main() -> None:
    if not BASE_INPUT.exists():
        raise FileNotFoundError(f"Base input not found:\n{BASE_INPUT}")

    if not DATAWARRIOR_EXPORT.exists():
        raise FileNotFoundError(
            "DataWarrior export file not found.\n"
            f"Expected file:\n{DATAWARRIOR_EXPORT}\n\n"
            "Export the DataWarrior table as CSV and save it with this exact name."
        )

    base = pd.read_csv(BASE_INPUT, encoding="utf-8-sig")
    dw = read_csv_flexible(DATAWARRIOR_EXPORT)

    if "Compound_ID" not in dw.columns:
        raise ValueError(
            "The DataWarrior export must contain a Compound_ID column.\n"
            f"Available columns are:\n{list(dw.columns)}"
        )

    clogp_col = find_clogp_column(dw)

    # Keep only needed columns and standardise name.
    dw_small = dw[["Compound_ID", clogp_col]].copy()
    dw_small = dw_small.rename(columns={clogp_col: "DataWarrior_cLogP"})

    # Decimal comma safety.
    dw_small["DataWarrior_cLogP"] = (
        dw_small["DataWarrior_cLogP"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    dw_small["DataWarrior_cLogP"] = pd.to_numeric(
        dw_small["DataWarrior_cLogP"],
        errors="coerce",
    )

    # Validate duplicates.
    if dw_small["Compound_ID"].duplicated().any():
        duplicates = dw_small.loc[
            dw_small["Compound_ID"].duplicated(),
            "Compound_ID"
        ].tolist()
        raise ValueError(f"Duplicate Compound_ID values in DataWarrior export: {duplicates[:10]}")

    # Merge against base input to preserve frozen benchmark order.
    merged = base.merge(
        dw_small,
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )

    missing_values = merged["DataWarrior_cLogP"].isna().sum()
    if missing_values > 0:
        missing_ids = merged.loc[
            merged["DataWarrior_cLogP"].isna(),
            "Compound_ID"
        ].tolist()
        raise ValueError(
            f"Missing DataWarrior cLogP values for {missing_values} compounds.\n"
            f"First missing IDs: {missing_ids[:20]}"
        )

    # Calculate errors.
    merged["logP_exp"] = pd.to_numeric(merged["logP_exp"], errors="raise")
    merged["N_count"] = pd.to_numeric(merged["N_count"], errors="raise")

    merged["delta_DataWarrior_cLogP"] = (
        merged["logP_exp"] - merged["DataWarrior_cLogP"]
    )
    merged["abs_delta_DataWarrior_cLogP"] = merged["delta_DataWarrior_cLogP"].abs()

    out = merged[
        [
            "Compound_ID",
            "SMILES",
            "logP_exp",
            "N_count",
            "FM",
            "DataWarrior_cLogP",
            "delta_DataWarrior_cLogP",
            "abs_delta_DataWarrior_cLogP",
        ]
    ].copy()

    OUT_VALUES.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_VALUES, index=False, encoding="utf-8")

    # Performance summary.
    y_true = out["logP_exp"].astype(float)
    y_pred = out["DataWarrior_cLogP"].astype(float)
    delta = out["delta_DataWarrior_cLogP"].astype(float)

    bias = float(delta.mean())
    mae = float(delta.abs().mean())
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    r2 = calculate_r2(y_true, y_pred)
    overestimated_percent = float((delta < 0).mean() * 100)
    severe_abs_error_percent = float((delta.abs() >= 2.0).mean() * 100)

    polar_mask = out["logP_exp"] <= 1.0
    high_risk_mask = out["N_count"].between(1, 3) & (out["logP_exp"] < 1.5)

    polar_bias = float(out.loc[polar_mask, "delta_DataWarrior_cLogP"].mean())
    high_risk_bias = float(out.loc[high_risk_mask, "delta_DataWarrior_cLogP"].mean())

    report = []
    report.append("Dataset_S42b DataWarrior cLogP processing report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Base input file: {BASE_INPUT}")
    report.append(f"DataWarrior export file: {DATAWARRIOR_EXPORT}")
    report.append(f"Detected cLogP column: {clogp_col}")
    report.append(f"Output values: {OUT_VALUES}")
    report.append(f"Output report: {OUT_REPORT}")
    report.append("")
    report.append("1. Processing status")
    report.append("-" * 70)
    report.append(f"Base compounds: {len(base)}")
    report.append(f"DataWarrior export rows: {len(dw)}")
    report.append(f"Valid DataWarrior cLogP values: {len(out)}")
    report.append(f"Missing DataWarrior cLogP values: {missing_values}")
    report.append("")
    report.append("2. DataWarrior cLogP performance summary")
    report.append("-" * 70)
    report.append(f"Bias: {bias:.3f}")
    report.append(f"MAE: {mae:.3f}")
    report.append(f"RMSE: {rmse:.3f}")
    report.append(f"R2: {r2:.3f}")
    report.append(f"Overestimated compounds: {overestimated_percent:.1f}%")
    report.append(f"Severe absolute error compounds: {severe_abs_error_percent:.1f}%")
    report.append(f"Polar logP <= 1.0 bias: {polar_bias:.3f}")
    report.append(f"N = 1-3 and logP < 1.5 bias: {high_risk_bias:.3f}")
    report.append("")
    report.append("3. Interpretation note")
    report.append("-" * 70)
    report.append(
        "DataWarrior cLogP is included as an external platform-independent "
        "comparator. It is not used to train, recalibrate, or replace any "
        "SwissADME-associated predictor. Its role is to test whether the "
        "observed prediction-error architecture is restricted to the SwissADME "
        "suite or also appears in an independent cLogP implementation."
    )

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("SUCCESS: DataWarrior cLogP processing completed.")


if __name__ == "__main__":
    main()