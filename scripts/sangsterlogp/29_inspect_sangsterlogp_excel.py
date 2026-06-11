# -*- coding: utf-8 -*-
"""
Inspect SangsterLogP Excel workbook before coumarin filtering.

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python scripts\29_inspect_sangsterlogp_excel.py
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")
INPUT_XLSX = Path(r"C:\Users\Ahmet Gunes\Downloads\Datasets.xlsx")

OUT_DIR = PROJECT_DIR / "data" / "external" / "SangsterLogP"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT = OUT_DIR / "SangsterLogP_workbook_inspection_report.txt"
SHEET_SUMMARY_CSV = OUT_DIR / "SangsterLogP_sheet_summary.csv"


def norm(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


smiles_keys = ["smiles", "canonicalsmiles", "isomericsmiles", "structure", "qsarsmiles"]
logp_keys = ["logp", "exp_logp", "experimental_logp", "logpexp", "expvalue", "value"]
id_keys = ["name", "compound", "compoundid", "id", "cas", "dtxsid", "inchikey", "inchi"]


if not INPUT_XLSX.exists():
    raise FileNotFoundError(f"Input Excel file not found:\n{INPUT_XLSX}")

xl = pd.ExcelFile(INPUT_XLSX)

summary_rows = []
report_lines = []

report_lines.append("SangsterLogP workbook inspection report")
report_lines.append("=" * 72)
report_lines.append(f"Input file: {INPUT_XLSX}")
report_lines.append(f"Detected sheets: {len(xl.sheet_names)}")
report_lines.append("")

for sheet in xl.sheet_names:
    try:
        df_head = pd.read_excel(INPUT_XLSX, sheet_name=sheet, nrows=10)
        df_full_shape = pd.read_excel(INPUT_XLSX, sheet_name=sheet, nrows=0)
        columns = list(df_head.columns)
    except Exception as e:
        report_lines.append(f"[ERROR] Sheet {sheet}: {e}")
        continue

    ncols = len(columns)

    # Estimate rows cheaply by reading only first column when possible
    try:
        df_one_col = pd.read_excel(INPUT_XLSX, sheet_name=sheet, usecols=[0])
        nrows = len(df_one_col)
    except Exception:
        nrows = "unknown"

    normalized_cols = {norm(c): c for c in columns}

    likely_smiles = []
    likely_logp = []
    likely_ids = []

    for key, original in normalized_cols.items():
        if any(k in key for k in smiles_keys):
            likely_smiles.append(original)
        if any(k in key for k in logp_keys):
            likely_logp.append(original)
        if any(k in key for k in id_keys):
            likely_ids.append(original)

    summary_rows.append({
        "sheet": sheet,
        "rows_estimated": nrows,
        "columns": ncols,
        "likely_smiles_columns": "; ".join(map(str, likely_smiles)),
        "likely_logp_columns": "; ".join(map(str, likely_logp)),
        "likely_id_columns": "; ".join(map(str, likely_ids)),
    })

    report_lines.append(f"Sheet: {sheet}")
    report_lines.append("-" * 72)
    report_lines.append(f"Estimated rows: {nrows}")
    report_lines.append(f"Columns: {ncols}")
    report_lines.append(f"Likely SMILES columns: {likely_smiles if likely_smiles else 'None detected'}")
    report_lines.append(f"Likely logP columns: {likely_logp if likely_logp else 'None detected'}")
    report_lines.append(f"Likely ID columns: {likely_ids if likely_ids else 'None detected'}")
    report_lines.append("")
    report_lines.append("First columns:")
    for c in columns[:40]:
        report_lines.append(f"  - {c}")
    if len(columns) > 40:
        report_lines.append(f"  ... plus {len(columns) - 40} more columns")
    report_lines.append("")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SHEET_SUMMARY_CSV, index=False, encoding="utf-8-sig")

REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

print("SangsterLogP workbook inspection completed.")
print(f"Report: {REPORT_TXT}")
print(f"Sheet summary CSV: {SHEET_SUMMARY_CSV}")
print("")
print(summary_df.to_string(index=False))