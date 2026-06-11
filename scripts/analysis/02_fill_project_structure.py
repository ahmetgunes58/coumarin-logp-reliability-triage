# -*- coding: utf-8 -*-
"""
Fill coumarin-logp project structure with final/canonical copies.

This script:
- creates clean project subfolders,
- copies existing generated files into the correct folders with final names,
- does not move, delete, or modify original files,
- writes a copy log for verification.
"""

from pathlib import Path
import shutil
import pandas as pd


ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp")
PROJECT = ROOT / "coumarin-logp"

LOG_DIR = ROOT / "_repository_prep"
LOG_DIR.mkdir(parents=True, exist_ok=True)
COPY_LOG = LOG_DIR / "project_fill_copy_log.csv"


# ---------------------------------------------------------------------
# Folder structure
# ---------------------------------------------------------------------

folders = [
    PROJECT / "manuscript",
    PROJECT / "supporting_information",
    PROJECT / "data" / "raw",
    PROJECT / "data" / "processed",
    PROJECT / "figures" / "main",
    PROJECT / "figures" / "supporting",
    PROJECT / "scripts",
    PROJECT / "dft",
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_first(patterns):
    """
    Search from ROOT and return the newest matching file.
    patterns: list of filename fragments.
    """
    candidates = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        hay = str(p).lower()
        if all(fragment.lower() in hay for fragment in patterns):
            candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_if_found(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source is None:
        return {
            "destination": str(destination),
            "source": "",
            "status": "MISSING_SOURCE",
        }

    try:
        shutil.copy2(source, destination)
        return {
            "destination": str(destination),
            "source": str(source),
            "status": "COPIED",
        }
    except Exception as exc:
        return {
            "destination": str(destination),
            "source": str(source),
            "status": f"ERROR: {exc}",
        }


log_rows = []


# ---------------------------------------------------------------------
# Data files: raw and processed
# ---------------------------------------------------------------------

copy_tasks = [
    # Raw / provenance
    (
        find_first(["raw_dataset_original.xlsx"]),
        PROJECT / "data" / "raw" / "Dataset_S0_raw_literature_collection.xlsx",
    ),

    # Final benchmark and processed outputs
    (
        PROJECT / "data" / "processed" / "benchmark_dataset.csv",
        PROJECT / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv",
    ),
    (
        PROJECT / "data" / "processed" / "Dataset_S4_structural_class_summary.csv",
        PROJECT / "data" / "processed" / "Dataset_S4_structural_class_summary.csv",
    ),
    (
        PROJECT / "data" / "processed" / "table1_overall_performance.csv",
        PROJECT / "data" / "processed" / "Dataset_S5_overall_predictor_performance.csv",
    ),
    (
        PROJECT / "data" / "processed" / "table2_nitrogen_bias.csv",
        PROJECT / "data" / "processed" / "Dataset_S6_nitrogen_group_bias.csv",
    ),
    (
        PROJECT / "data" / "processed" / "table4_logp_range.csv",
        PROJECT / "data" / "processed" / "Dataset_S7_logP_range_performance.csv",
    ),
    (
        PROJECT / "data" / "processed" / "table4b_regression.csv",
        PROJECT / "data" / "processed" / "Dataset_S8_regression_outputs.csv",
    ),
    (
        PROJECT / "data" / "processed" / "table4c_2d_heatmap_bias.csv",
        PROJECT / "data" / "processed" / "Dataset_S9_2D_risk_map_bias.csv",
    ),
    (
        PROJECT / "data" / "processed" / "table4c_2d_heatmap_counts.csv",
        PROJECT / "data" / "processed" / "Dataset_S10_2D_risk_map_counts.csv",
    ),

    # Figure 6 source data/statistics
    (
        find_first(["Figure6_source_data_v2.csv"]),
        PROJECT / "data" / "processed" / "Dataset_S11_predictor_disagreement_Spread4.csv",
    ),
    (
        find_first(["Figure6_statistics_v2.txt"]),
        PROJECT / "data" / "processed" / "Dataset_S11_predictor_disagreement_statistics.txt",
    ),

    # FM taxonomy summary
    (
        find_first(["Table10_taxonomy_summary_FINAL.csv"]),
        PROJECT / "data" / "processed" / "Dataset_S12_failure_mode_taxonomy_summary.csv",
    ),

    # DFT final table values
    (
        find_first(["dft_panel_values_final.csv"]),
        PROJECT / "data" / "processed" / "Dataset_S13_DFT_panel_values.csv",
    ),
]


# ---------------------------------------------------------------------
# Main figures already generated
# ---------------------------------------------------------------------

copy_tasks.extend([
    (
        find_first(["Figure6_predictor_disagreement_v2.png"]),
        PROJECT / "figures" / "main" / "Figure_6_predictor_disagreement.png",
    ),
    (
        find_first(["Figure6_predictor_disagreement_v2.tiff"]),
        PROJECT / "figures" / "main" / "Figure_6_predictor_disagreement.tiff",
    ),
    (
        find_first(["Figure6_predictor_disagreement_v2.pdf"]),
        PROJECT / "figures" / "main" / "Figure_6_predictor_disagreement.pdf",
    ),
    (
        find_first(["Figure8_failure_mode_taxonomy_FINAL.png"]),
        PROJECT / "figures" / "main" / "Figure_8_failure_mode_taxonomy.png",
    ),
    (
        find_first(["Figure8_failure_mode_taxonomy_FINAL.tiff"]),
        PROJECT / "figures" / "main" / "Figure_8_failure_mode_taxonomy.tiff",
    ),
    (
        find_first(["Figure8_failure_mode_taxonomy_FINAL.pdf"]),
        PROJECT / "figures" / "main" / "Figure_8_failure_mode_taxonomy.pdf",
    ),
])


# ---------------------------------------------------------------------
# Supporting information document
# ---------------------------------------------------------------------

copy_tasks.extend([
    (
        find_first(["SI_Template_coumarin-logp_Coumarin_LogP", ".docx"]),
        PROJECT / "supporting_information" / "Supporting_Information_coumarin-logp_Coumarin_LogP.docx",
    )
])


# ---------------------------------------------------------------------
# Scripts: copy currently prepared project scripts
# ---------------------------------------------------------------------

copy_tasks.extend([
    (
        ROOT / "00_project_inventory.py",
        PROJECT / "scripts" / "00_project_inventory.py",
    ),
    (
        ROOT / "01_repository_candidate_report.py",
        PROJECT / "scripts" / "01_repository_candidate_report.py",
    ),
    (
        ROOT / "02_fill_project_structure.py",
        PROJECT / "scripts" / "02_fill_project_structure.py",
    ),
])


# ---------------------------------------------------------------------
# Execute copy tasks
# ---------------------------------------------------------------------

for source, destination in copy_tasks:
    log_rows.append(copy_if_found(source, destination))


log_df = pd.DataFrame(log_rows)
log_df.to_csv(COPY_LOG, index=False, encoding="utf-8-sig")


print("\nProject structure filled with canonical copies.")
print(f"Project folder: {PROJECT}")
print(f"Copy log: {COPY_LOG}")

print("\nCopy status summary:")
print(log_df["status"].value_counts().to_string())

print("\nMissing sources:")
missing = log_df[log_df["status"] == "MISSING_SOURCE"]
if missing.empty:
    print("None")
else:
    print(missing[["destination"]].to_string(index=False))