# -*- coding: utf-8 -*-
"""
00_prepare_external_predictor_input.py

Purpose
-------
Prepare the input files for the external platform-independent logP comparator audit.

This script uses the frozen benchmark dataset:
    data/processed/Dataset_S1_benchmark_dataset.csv

It creates:
    data/processed/Dataset_S42_external_predictor_input.csv
    data/external_predictor_input.smi
    data/external_predictor_input.tsv
    data/processed/Dataset_S42_input_validation_report.txt

Important
---------
This script does NOT modify Dataset_S1_benchmark_dataset.csv.
Dataset_S1 remains the frozen benchmark dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This script is expected to be saved in:
# coumarin-logp/scripts/00_prepare_external_predictor_input.py
# Therefore, project root is one level above /scripts.
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_S1 = PROJECT_DIR / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

OUT_PROCESSED = PROJECT_DIR / "data" / "processed"
OUT_DATA = PROJECT_DIR / "data"

OUT_EXTERNAL_INPUT = OUT_PROCESSED / "Dataset_S42_external_predictor_input.csv"
OUT_SMI = OUT_DATA / "external_predictor_input.smi"
OUT_TSV = OUT_DATA / "external_predictor_input.tsv"
OUT_REPORT = OUT_PROCESSED / "Dataset_S42_input_validation_report.txt"


# ---------------------------------------------------------------------
# Required columns for the external comparator audit
# ---------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Compound_ID",
    "SMILES",
    "logP_exp",
    "N_count",
    "N_group",
    "FM",
    "Data_Tier",
    "Consensus",
    "delta_Consensus",
]

OPTIONAL_COLUMNS = [
    "logP_range",
    "logP_15",
    "Coumarin_Type",
    "SMILES_valid",
]


def metric_rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def main() -> None:
    if not DATASET_S1.exists():
        raise FileNotFoundError(f"Dataset_S1 not found:\n{DATASET_S1}")

    OUT_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_S1, encoding="utf-8-sig")

    report = []
    report.append("Dataset_S42 external predictor input preparation report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Input file: {DATASET_S1}")
    report.append("")

    # -----------------------------------------------------------------
    # Basic structure checks
    # -----------------------------------------------------------------
    report.append("1. Basic dataset checks")
    report.append("-" * 70)
    report.append(f"Number of rows: {len(df)}")
    report.append(f"Number of columns: {len(df.columns)}")

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    duplicate_ids = df["Compound_ID"].duplicated().sum()
    report.append(f"Duplicate Compound_ID values: {duplicate_ids}")

    missing_core = df[["Compound_ID", "SMILES", "logP_exp"]].isna().sum()
    report.append("Missing values in core columns:")
    for col, val in missing_core.items():
        report.append(f"  {col}: {int(val)}")

    if duplicate_ids > 0:
        dup_values = df.loc[df["Compound_ID"].duplicated(), "Compound_ID"].tolist()
        raise ValueError(f"Duplicate Compound_ID values detected: {dup_values[:10]}")

    if missing_core.sum() > 0:
        raise ValueError("Missing values detected in Compound_ID, SMILES, or logP_exp.")

    # -----------------------------------------------------------------
    # Numeric checks
    # -----------------------------------------------------------------
    df["logP_exp"] = pd.to_numeric(df["logP_exp"], errors="raise")
    df["N_count"] = pd.to_numeric(df["N_count"], errors="raise")
    df["Consensus"] = pd.to_numeric(df["Consensus"], errors="raise")
    df["delta_Consensus"] = pd.to_numeric(df["delta_Consensus"], errors="raise")

    recalculated_delta = df["logP_exp"] - df["Consensus"]
    max_delta_difference = float((recalculated_delta - df["delta_Consensus"]).abs().max())

    report.append("")
    report.append("2. Numerical consistency checks")
    report.append("-" * 70)
    report.append(f"logP_exp minimum: {df['logP_exp'].min():.4f}")
    report.append(f"logP_exp maximum: {df['logP_exp'].max():.4f}")
    report.append(
        "Max absolute difference between stored delta_Consensus and "
        f"recalculated logP_exp - Consensus: {max_delta_difference:.10f}"
    )

    if max_delta_difference > 1e-6:
        raise ValueError(
            "delta_Consensus is not consistent with logP_exp - Consensus. "
            f"Maximum difference: {max_delta_difference}"
        )

    # -----------------------------------------------------------------
    # Expected dataset summaries
    # -----------------------------------------------------------------
    report.append("")
    report.append("3. Dataset summaries")
    report.append("-" * 70)

    if "Data_Tier" in df.columns:
        report.append("Data_Tier distribution:")
        for key, val in df["Data_Tier"].value_counts().sort_index().items():
            report.append(f"  {key}: {int(val)}")

    if "N_group" in df.columns:
        report.append("N_group distribution:")
        for key, val in df["N_group"].value_counts().sort_index().items():
            report.append(f"  {key}: {int(val)}")

    if "FM" in df.columns:
        report.append("FM distribution:")
        for key, val in df["FM"].value_counts().sort_index().items():
            report.append(f"  {key}: {int(val)}")

    # -----------------------------------------------------------------
    # Recalculate key manuscript values for validation
    # -----------------------------------------------------------------
    delta = df["delta_Consensus"]
    y_true = df["logP_exp"]
    y_pred = df["Consensus"]

    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    consensus_bias = float(delta.mean())
    consensus_mae = float(delta.abs().mean())
    consensus_rmse = metric_rmse(delta)
    overestimated_percent = float((delta < 0).mean() * 100)
    severe_abs_error_percent = float((delta.abs() >= 2.0).mean() * 100)

    polar_mask = df["logP_exp"] <= 1.0
    high_risk_mask = df["N_count"].between(1, 3) & (df["logP_exp"] < 1.5)

    report.append("")
    report.append("4. Recalculated key manuscript values")
    report.append("-" * 70)
    report.append(f"Consensus bias: {consensus_bias:.3f}")
    report.append(f"Consensus MAE: {consensus_mae:.3f}")
    report.append(f"Consensus RMSE: {consensus_rmse:.3f}")
    report.append(f"Consensus R2: {r2:.3f}")
    report.append(f"Overestimated compounds: {overestimated_percent:.1f}%")
    report.append(f"Severe absolute error compounds: {severe_abs_error_percent:.1f}%")
    report.append(
        f"Polar logP <= 1.0 consensus bias: {df.loc[polar_mask, 'delta_Consensus'].mean():.3f}"
    )
    report.append(
        "N = 1-3 and logP < 1.5 consensus bias: "
        f"{df.loc[high_risk_mask, 'delta_Consensus'].mean():.3f}"
    )

    # -----------------------------------------------------------------
    # DFT panel presence check
    # -----------------------------------------------------------------
    expected_dft_panel = [
        "CMR_GOLD_055",
        "CMR_GOLD_043",
        "CMR_GOLD_044",
        "CMR_GOLD_029",
        "CMR_GOLD_058",
        "CMR_GOLD_079",
        "CMR_GOLD_016",
        "CMR_GOLD_090",
        "CMR_GOLD_020",
        "CMR_GOLD_092",
    ]

    present_ids = set(df["Compound_ID"])
    missing_dft = [cid for cid in expected_dft_panel if cid not in present_ids]

    report.append("")
    report.append("5. Ten-compound DFT panel presence check")
    report.append("-" * 70)
    if missing_dft:
        report.append(f"Missing DFT-panel compounds: {missing_dft}")
        raise ValueError(f"Missing DFT-panel compounds: {missing_dft}")
    else:
        report.append("All ten DFT-panel compounds are present in Dataset_S1.")

    # -----------------------------------------------------------------
    # Prepare external predictor input files
    # -----------------------------------------------------------------
    selected_columns = REQUIRED_COLUMNS + [
        col for col in OPTIONAL_COLUMNS if col in df.columns
    ]

    external_input = df[selected_columns].copy()
    external_input.to_csv(OUT_EXTERNAL_INPUT, index=False, encoding="utf-8")

    # SMILES file: SMILES<TAB>Compound_ID
    with OUT_SMI.open("w", encoding="utf-8", newline="") as f:
        for _, row in df.iterrows():
            f.write(f"{row['SMILES']}\t{row['Compound_ID']}\n")

    # TSV file for DataWarrior or manual import
    tsv_columns = ["Compound_ID", "SMILES", "logP_exp", "N_count", "FM"]
    df[tsv_columns].to_csv(OUT_TSV, sep="\t", index=False, encoding="utf-8")

    report.append("")
    report.append("6. Output files created")
    report.append("-" * 70)
    report.append(str(OUT_EXTERNAL_INPUT))
    report.append(str(OUT_SMI))
    report.append(str(OUT_TSV))
    report.append(str(OUT_REPORT))

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("SUCCESS: External predictor input preparation completed.")


if __name__ == "__main__":
    main()