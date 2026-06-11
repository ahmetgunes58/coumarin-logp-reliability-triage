# -*- coding: utf-8 -*-
"""
06_inspect_opera_raw_output.py

Purpose
-------
Inspect OPERA raw output before standardising OPERA logP values.

Input
-----
data/processed/OPERA_logP_raw_output.csv

Output
------
data/processed/Dataset_S42d_OPERA_raw_output_structure_report.txt

This script does not calculate performance metrics. It only reports:
- detected separator
- number of rows and columns
- column names
- first rows
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_OUTPUT = PROJECT_DIR / "data" / "processed" / "OPERA_logP_raw_output.csv"
OUT_REPORT = PROJECT_DIR / "data" / "processed" / "Dataset_S42d_OPERA_raw_output_structure_report.txt"


def read_flexible(path: Path):
    attempts = []

    for sep_name, sep in [
        ("comma", ","),
        ("tab", "\t"),
        ("semicolon", ";"),
    ]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            attempts.append((sep_name, sep, df))
        except Exception:
            pass

    if not attempts:
        raise ValueError(f"Could not read OPERA output file: {path}")

    # Choose parse with largest number of columns.
    sep_name, sep, df = max(attempts, key=lambda x: len(x[2].columns))
    return sep_name, sep, df


def main() -> None:
    if not RAW_OUTPUT.exists():
        raise FileNotFoundError(f"OPERA raw output not found:\n{RAW_OUTPUT}")

    sep_name, sep, df = read_flexible(RAW_OUTPUT)

    report = []
    report.append("OPERA raw output structure report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Raw OPERA output: {RAW_OUTPUT}")
    report.append(f"Detected separator: {sep_name!r} ({sep!r})")
    report.append("")
    report.append("1. Shape")
    report.append("-" * 70)
    report.append(f"Rows: {len(df)}")
    report.append(f"Columns: {len(df.columns)}")
    report.append("")
    report.append("2. Column names")
    report.append("-" * 70)
    for i, col in enumerate(df.columns, start=1):
        report.append(f"{i:02d}. {col}")
    report.append("")
    report.append("3. First 10 rows")
    report.append("-" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        report.append(df.head(10).to_string(index=False))

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("SUCCESS: OPERA raw output inspection completed.")


if __name__ == "__main__":
    main()