# -*- coding: utf-8 -*-
"""
04_process_alogps_2_1.py

Purpose
-------
Parse ALOGPS 2.1 raw web output and standardise it for the external
platform-independent logP comparator audit.

Input
-----
data/processed/ALOGPS_2_1_raw_output.txt
data/processed/Dataset_S42c_ALOGPS_ordered_input_map.csv

Outputs
-------
data/processed/Dataset_S42c_ALOGPS_2_1_values.csv
data/processed/Dataset_S42c_ALOGPS_2_1_report.txt

Important
---------
ALOGPS output usually labels molecules as mol_1, mol_2, ..., mol_n.
These are mapped back to Compound_ID using the preserved input order.
The second numeric column in the ALOGPS output is interpreted as ALOGPS logP.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]

ORDER_MAP = PROJECT_DIR / "data" / "processed" / "Dataset_S42c_ALOGPS_ordered_input_map.csv"
RAW_OUTPUT = PROJECT_DIR / "data" / "processed" / "ALOGPS_2_1_raw_output.txt"

OUT_VALUES = PROJECT_DIR / "data" / "processed" / "Dataset_S42c_ALOGPS_2_1_values.csv"
OUT_REPORT = PROJECT_DIR / "data" / "processed" / "Dataset_S42c_ALOGPS_2_1_report.txt"


LINE_PATTERN = re.compile(
    r"^\s*mol[_\s-]*(\d+)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"(.+?)\s*$",
    re.IGNORECASE,
)


def calculate_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def parse_alogps_raw_output(raw_text: str) -> pd.DataFrame:
    rows = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = LINE_PATTERN.match(line)
        if not match:
            continue

        order = int(match.group(1))
        logp = float(match.group(2))
        logs = float(match.group(3))
        smiles_out = match.group(4).strip()

        rows.append(
            {
                "ALOGPS_order": order,
                "ALOGPS_2_1_logP": logp,
                "ALOGPS_logS": logs,
                "ALOGPS_output_SMILES": smiles_out,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not ORDER_MAP.exists():
        raise FileNotFoundError(f"Order map not found:\n{ORDER_MAP}")

    if not RAW_OUTPUT.exists():
        raise FileNotFoundError(
            "ALOGPS raw output file not found.\n"
            f"Expected file:\n{RAW_OUTPUT}\n\n"
            "Copy the full ALOGPS result page into this text file."
        )

    order_map = pd.read_csv(ORDER_MAP, encoding="utf-8-sig")
    raw_text = RAW_OUTPUT.read_text(encoding="utf-8", errors="replace")

    parsed = parse_alogps_raw_output(raw_text)

    if parsed.empty:
        raise ValueError(
            "No ALOGPS result lines were parsed.\n"
            "Expected lines like: mol_1 3.06 -4.45 SMILES"
        )

    if parsed["ALOGPS_order"].duplicated().any():
        duplicates = parsed.loc[
            parsed["ALOGPS_order"].duplicated(),
            "ALOGPS_order"
        ].tolist()
        raise ValueError(f"Duplicate ALOGPS mol/order entries detected: {duplicates[:20]}")

    expected_n = len(order_map)
    parsed_n = len(parsed)

    if parsed_n != expected_n:
        missing_orders = sorted(set(order_map["ALOGPS_order"]) - set(parsed["ALOGPS_order"]))
        extra_orders = sorted(set(parsed["ALOGPS_order"]) - set(order_map["ALOGPS_order"]))

        raise ValueError(
            f"ALOGPS parsed row count mismatch.\n"
            f"Expected: {expected_n}\n"
            f"Parsed: {parsed_n}\n"
            f"Missing orders: {missing_orders[:30]}\n"
            f"Extra orders: {extra_orders[:30]}\n\n"
            "Please make sure the raw output file contains all mol_1 to mol_95 result lines."
        )

    merged = order_map.merge(
        parsed,
        on="ALOGPS_order",
        how="left",
        validate="one_to_one",
    )

    if merged["ALOGPS_2_1_logP"].isna().any():
        missing_ids = merged.loc[
            merged["ALOGPS_2_1_logP"].isna(),
            "Compound_ID"
        ].tolist()
        raise ValueError(f"Missing ALOGPS logP values for: {missing_ids[:20]}")

    merged["logP_exp"] = pd.to_numeric(merged["logP_exp"], errors="raise")
    merged["N_count"] = pd.to_numeric(merged["N_count"], errors="raise")

    merged["delta_ALOGPS_2_1_logP"] = (
        merged["logP_exp"] - merged["ALOGPS_2_1_logP"]
    )
    merged["abs_delta_ALOGPS_2_1_logP"] = merged["delta_ALOGPS_2_1_logP"].abs()

    out = merged[
        [
            "ALOGPS_order",
            "Compound_ID",
            "SMILES",
            "logP_exp",
            "N_count",
            "FM",
            "ALOGPS_2_1_logP",
            "ALOGPS_logS",
            "delta_ALOGPS_2_1_logP",
            "abs_delta_ALOGPS_2_1_logP",
            "ALOGPS_output_SMILES",
        ]
    ].copy()

    OUT_VALUES.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_VALUES, index=False, encoding="utf-8")

    # -----------------------------------------------------------------
    # Performance summary
    # -----------------------------------------------------------------
    y_true = out["logP_exp"].astype(float)
    y_pred = out["ALOGPS_2_1_logP"].astype(float)
    delta = out["delta_ALOGPS_2_1_logP"].astype(float)

    bias = float(delta.mean())
    mae = float(delta.abs().mean())
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    r2 = calculate_r2(y_true, y_pred)
    overestimated_percent = float((delta < 0).mean() * 100)
    severe_abs_error_percent = float((delta.abs() >= 2.0).mean() * 100)

    polar_mask = out["logP_exp"] <= 1.0
    high_risk_mask = out["N_count"].between(1, 3) & (out["logP_exp"] < 1.5)

    polar_bias = float(out.loc[polar_mask, "delta_ALOGPS_2_1_logP"].mean())
    high_risk_bias = float(out.loc[high_risk_mask, "delta_ALOGPS_2_1_logP"].mean())

    report = []
    report.append("Dataset_S42c ALOGPS 2.1 processing report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Order map: {ORDER_MAP}")
    report.append(f"Raw ALOGPS output: {RAW_OUTPUT}")
    report.append(f"Output values: {OUT_VALUES}")
    report.append(f"Output report: {OUT_REPORT}")
    report.append("")
    report.append("1. Processing status")
    report.append("-" * 70)
    report.append(f"Expected compounds: {expected_n}")
    report.append(f"Parsed ALOGPS result lines: {parsed_n}")
    report.append(f"Valid ALOGPS logP values: {out['ALOGPS_2_1_logP'].notna().sum()}")
    report.append(f"Parsed order minimum: {int(parsed['ALOGPS_order'].min())}")
    report.append(f"Parsed order maximum: {int(parsed['ALOGPS_order'].max())}")
    report.append("")
    report.append("2. ALOGPS 2.1 performance summary")
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
        "ALOGPS 2.1 is included as an external platform-independent comparator. "
        "It is not used to train, recalibrate, or replace any SwissADME-associated "
        "predictor. Its role is to test whether the observed prediction-error "
        "architecture is restricted to the SwissADME suite or also appears in "
        "an independently parameterised logP implementation."
    )

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("SUCCESS: ALOGPS 2.1 processing completed.")


if __name__ == "__main__":
    main()