# -*- coding: utf-8 -*-
"""
03_prepare_alogps_input.py

Purpose
-------
Prepare ordered input files for ALOGPS 2.1 external logP comparator audit.

Input
-----
data/processed/Dataset_S42_external_predictor_input.csv

Outputs
-------
data/processed/Dataset_S42c_ALOGPS_ordered_input_map.csv
data/ALOGPS_2_1_input_smiles_only.txt
data/ALOGPS_2_1_input_smiles_with_id.txt
data/processed/Dataset_S42c_ALOGPS_2_1_template.csv
data/processed/Dataset_S42c_ALOGPS_2_1_input_report.txt

Important
---------
ALOGPS web output may not preserve Compound_ID explicitly.
Therefore, the compound order is preserved and documented.
If the ALOGPS output is copied in the same order, it can be merged safely by row order.
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S42_external_predictor_input.csv"

OUT_ORDER_MAP = PROJECT_DIR / "data" / "processed" / "Dataset_S42c_ALOGPS_ordered_input_map.csv"
OUT_SMILES_ONLY = PROJECT_DIR / "data" / "ALOGPS_2_1_input_smiles_only.txt"
OUT_SMILES_WITH_ID = PROJECT_DIR / "data" / "ALOGPS_2_1_input_smiles_with_id.txt"
OUT_TEMPLATE = PROJECT_DIR / "data" / "processed" / "Dataset_S42c_ALOGPS_2_1_template.csv"
OUT_REPORT = PROJECT_DIR / "data" / "processed" / "Dataset_S42c_ALOGPS_2_1_input_report.txt"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required = ["Compound_ID", "SMILES", "logP_exp", "N_count", "FM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["Compound_ID"].duplicated().any():
        raise ValueError("Duplicate Compound_ID values detected.")

    if df[["Compound_ID", "SMILES", "logP_exp"]].isna().any().any():
        raise ValueError("Missing values detected in Compound_ID, SMILES, or logP_exp.")

    df = df.copy()
    df.insert(0, "ALOGPS_order", range(1, len(df) + 1))

    # Order map for reproducibility.
    order_map = df[
        ["ALOGPS_order", "Compound_ID", "SMILES", "logP_exp", "N_count", "FM"]
    ].copy()
    order_map.to_csv(OUT_ORDER_MAP, index=False, encoding="utf-8")

    # File 1: SMILES only. This is the safest for web batch input.
    with OUT_SMILES_ONLY.open("w", encoding="utf-8", newline="") as f:
        for _, row in df.iterrows():
            f.write(f"{row['SMILES']}\n")

    # File 2: SMILES + ID. Use this only if ALOGPS accepts names after SMILES.
    with OUT_SMILES_WITH_ID.open("w", encoding="utf-8", newline="") as f:
        for _, row in df.iterrows():
            f.write(f"{row['SMILES']}\t{row['Compound_ID']}\n")

    # Template to paste ALOGPS output manually.
    template = df[
        ["ALOGPS_order", "Compound_ID", "SMILES", "logP_exp", "N_count", "FM"]
    ].copy()
    template["ALOGPS_2_1_logP"] = ""
    template["ALOGPS_notes"] = ""
    template.to_csv(OUT_TEMPLATE, index=False, encoding="utf-8")

    report = []
    report.append("Dataset_S42c ALOGPS 2.1 input preparation report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Input file: {INPUT_FILE}")
    report.append("")
    report.append("Output files:")
    report.append(f"- {OUT_ORDER_MAP}")
    report.append(f"- {OUT_SMILES_ONLY}")
    report.append(f"- {OUT_SMILES_WITH_ID}")
    report.append(f"- {OUT_TEMPLATE}")
    report.append(f"- {OUT_REPORT}")
    report.append("")
    report.append(f"Number of compounds prepared for ALOGPS: {len(df)}")
    report.append("")
    report.append("Use instruction:")
    report.append("1. Use ALOGPS_2_1_input_smiles_only.txt for ALOGPS batch prediction.")
    report.append("2. Copy the ALOGPS 2.1 logP values into Dataset_S42c_ALOGPS_2_1_template.csv.")
    report.append("3. Save the filled file as Dataset_S42c_ALOGPS_2_1_template_filled.csv.")
    report.append("4. The next script will validate and process the filled ALOGPS values.")

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("SUCCESS: ALOGPS 2.1 input files prepared.")


if __name__ == "__main__":
    main()