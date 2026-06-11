# -*- coding: utf-8 -*-
"""
Copy verified DFT panel files into the coumarin-logp project DFT archive.

Input:
    data/processed/Dataset_S17a_orca_tests_output_candidates.csv

Output archive:
    dft/molecules/CMR_GOLD_XXX/01_opt_freq
    dft/molecules/CMR_GOLD_XXX/02_sp_charge

This script:
- copies, does not move,
- does not delete or modify source files,
- copies common ORCA-related files from each selected calculation folder,
- writes a copy log.
"""

from pathlib import Path
import shutil
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")

CANDIDATE_CSV = PROJECT / "data" / "processed" / "Dataset_S17a_orca_tests_output_candidates.csv"

DFT_PROJECT_ROOT = PROJECT / "dft"
DFT_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

COPY_LOG = PROJECT / "data" / "processed" / "Dataset_S17b_DFT_archive_copy_log.csv"


# ============================================================
# Configuration
# ============================================================

allowed_suffixes = {
    ".out",
    ".inp",
    ".xyz",
    ".gbw",
    ".molden",
    ".cube",
    ".cub",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".txt",
    ".log",
}

calc_folder_map = {
    "opt_freq": "01_opt_freq",
    "single_point": "02_sp_charge",
}


# ============================================================
# Load candidate report
# ============================================================

df = pd.read_csv(CANDIDATE_CSV)

required_cols = [
    "Compound",
    "Candidate_calc_type",
    "Full_path",
    "Normal_termination",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in candidate CSV: {missing}")

# Keep only normal-terminated opt_freq and single_point files
df = df[
    (df["Normal_termination"] == True)
    & (df["Candidate_calc_type"].isin(["opt_freq", "single_point"]))
].copy()

if df.empty:
    raise ValueError("No verified DFT candidate files found for copying.")


# ============================================================
# Copy folders
# ============================================================

log_rows = []

for _, row in df.iterrows():
    compound = row["Compound"]
    calc_type = row["Candidate_calc_type"]
    source_out = Path(row["Full_path"])

    if not source_out.exists():
        log_rows.append({
            "Compound": compound,
            "Calc_type": calc_type,
            "Source_file": str(source_out),
            "Destination_file": "",
            "Status": "SOURCE_OUT_NOT_FOUND",
        })
        continue

    source_dir = source_out.parent
    dest_dir = DFT_PROJECT_ROOT / "molecules" / compound / calc_folder_map[calc_type]
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_suffixes
    ]

    if not files_to_copy:
        log_rows.append({
            "Compound": compound,
            "Calc_type": calc_type,
            "Source_file": str(source_dir),
            "Destination_file": str(dest_dir),
            "Status": "NO_ALLOWED_FILES_IN_SOURCE_DIR",
        })
        continue

    for source_file in files_to_copy:
        dest_file = dest_dir / source_file.name

        try:
            shutil.copy2(source_file, dest_file)
            status = "COPIED"
        except Exception as exc:
            status = f"ERROR: {exc}"

        log_rows.append({
            "Compound": compound,
            "Calc_type": calc_type,
            "Source_file": str(source_file),
            "Destination_file": str(dest_file),
            "Status": status,
        })


# ============================================================
# Save log
# ============================================================

log_df = pd.DataFrame(log_rows)
log_df.to_csv(COPY_LOG, index=False, encoding="utf-8-sig")


print("\nVerified DFT panel files copied into project archive.")
print(f"Candidate report: {CANDIDATE_CSV}")
print(f"Project DFT archive: {DFT_PROJECT_ROOT}")
print(f"Copy log: {COPY_LOG}")

print("\nCopy status summary:")
print(log_df["Status"].value_counts().to_string())

print("\nCopied files by compound/calculation:")
print(
    log_df[log_df["Status"] == "COPIED"]
    .groupby(["Compound", "Calc_type"])
    .size()
    .to_string()
)