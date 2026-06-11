# -*- coding: utf-8 -*-
"""
Project inventory script for coumarin-logp coumarin logP manuscript.

Purpose:
- List all files under the project root.
- Categorise files as manuscript, SI, data, figure, script, DFT, log, etc.
- Create project_inventory.csv and project_tree.txt.
- Prepare a clean repository-planning overview before GitHub deposition.

This script does not move, rename, or delete any file.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp")
OUTDIR = ROOT / "_repository_prep"
OUTDIR.mkdir(parents=True, exist_ok=True)

INVENTORY_CSV = OUTDIR / "project_inventory.csv"
TREE_TXT = OUTDIR / "project_tree.txt"
SUMMARY_CSV = OUTDIR / "file_category_summary.csv"
REPO_STRUCTURE_TXT = OUTDIR / "recommended_repository_structure.txt"


def classify_file(path: Path) -> str:
    """Assign a broad category based on path and extension."""
    rel = str(path.relative_to(ROOT)).lower()
    ext = path.suffix.lower()

    if "_repository_prep" in rel:
        return "repository_prep"

    if ext in [".docx", ".doc", ".odt"]:
        if "si" in rel or "support" in rel:
            return "supporting_information_document"
        return "manuscript_document"

    if ext in [".pdf"]:
        if "si" in rel or "support" in rel:
            return "supporting_information_pdf"
        return "pdf_document"

    if ext in [".csv", ".xlsx", ".xls", ".tsv"]:
        if "raw" in rel:
            return "data_raw"
        if "processed" in rel or "table" in rel or "dataset" in rel:
            return "data_processed"
        return "data_other"

    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg", ".eps"]:
        if "figure" in rel or "figures" in rel:
            return "figure"
        if "mep" in rel or "dft" in rel:
            return "dft_visual"
        return "image_other"

    if ext in [".py", ".ipynb", ".r", ".R"]:
        return "script"

    if ext in [".inp", ".out", ".gbw", ".xyz", ".cube", ".molden", ".molden.input"]:
        return "dft_file"

    if ext in [".log", ".txt"]:
        if "orca" in rel or "dft" in rel:
            return "dft_log_or_text"
        return "log_or_text"

    if ext in [".zip", ".rar", ".7z"]:
        return "archive"

    return "other"


def file_record(path: Path) -> dict:
    stat = path.stat()
    rel = path.relative_to(ROOT)

    return {
        "relative_path": str(rel),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "category": classify_file(path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 4),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "parent_folder": str(rel.parent),
    }


def build_tree(paths):
    """Create a readable text tree from file paths."""
    lines = [str(ROOT), ""]
    sorted_paths = sorted(paths, key=lambda p: str(p).lower())

    for path in sorted_paths:
        rel = path.relative_to(ROOT)
        depth = len(rel.parts) - 1
        indent = "    " * depth
        marker = "├── "
        lines.append(f"{indent}{marker}{rel.name}")

    return "\n".join(lines)


def main():
    all_files = [p for p in ROOT.rglob("*") if p.is_file()]

    records = [file_record(p) for p in all_files]
    df = pd.DataFrame(records)

    df = df.sort_values(
        by=["category", "parent_folder", "filename"],
        ascending=[True, True, True]
    )

    df.to_csv(INVENTORY_CSV, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("category")
          .agg(
              n_files=("filename", "count"),
              total_size_mb=("size_mb", "sum")
          )
          .reset_index()
          .sort_values("category")
    )
    summary["total_size_mb"] = summary["total_size_mb"].round(3)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    TREE_TXT.write_text(build_tree(all_files), encoding="utf-8")

    repo_structure = """Recommended final GitHub / SI repository structure

coumarin-logp_Coumarin_logP/
├── README.md
├── CITATION.cff
├── manuscript/
│   ├── main_manuscript.docx
│   └── main_manuscript.pdf
├── supporting_information/
│   ├── Supporting_Information.docx
│   └── Supporting_Information.pdf
├── data/
│   ├── raw/
│   │   └── Dataset_S0_raw_literature_collection.xlsx
│   ├── processed/
│   │   ├── Dataset_S1_benchmark_dataset.csv
│   │   ├── Dataset_S2_experimental_sources.csv
│   │   ├── Dataset_S3_exclusion_log.csv
│   │   ├── Dataset_S4_structural_class_summary.csv
│   │   ├── Dataset_S5_overall_predictor_performance.csv
│   │   ├── Dataset_S6_nitrogen_group_bias.csv
│   │   ├── Dataset_S7_logP_range_performance.csv
│   │   ├── Dataset_S8_regression_outputs.csv
│   │   ├── Dataset_S9_2D_risk_map_bias.csv
│   │   ├── Dataset_S10_2D_risk_map_counts.csv
│   │   ├── Dataset_S11_predictor_disagreement_Spread4.csv
│   │   ├── Dataset_S12_failure_mode_taxonomy_summary.csv
│   │   ├── Dataset_S13_DFT_panel_values.csv
│   │   ├── Dataset_S14_DFT_Mulliken_charges.csv
│   │   ├── Dataset_S15_DFT_Lowdin_charges.csv
│   │   ├── Dataset_S16_DFT_calculation_status.csv
│   │   ├── Dataset_S17_DFT_file_manifest.csv
│   │   ├── Dataset_S18_six_predictor_scatter_source_data.csv
│   │   ├── Dataset_S19_six_predictor_error_source_data.csv
│   │   ├── Dataset_S20_expanded_ncount_logP_map_source_data.csv
│   │   └── Dataset_S21_repository_file_manifest.csv
├── figures/
│   ├── main/
│   │   ├── Figure_6_predictor_disagreement.png
│   │   ├── Figure_7_MEP_CMR_GOLD_043_058.png
│   │   └── Figure_8_failure_mode_taxonomy.png
│   └── supporting/
│       ├── Figure_S1_structural_class_distribution.png
│       ├── Figure_S2_six_predictor_scatter_plots.png
│       ├── Figure_S3_six_predictor_error_distributions.png
│       ├── Figure_S4_expanded_ncount_logP_map.png
│       └── Figure_S5_auxiliary_MEP_maps.png
├── scripts/
│   ├── Script_S1_prepare_benchmark_dataset.py
│   ├── Script_S2_reproduce_statistics.py
│   ├── Script_S3_assign_failure_modes.py
│   ├── Script_S4_make_main_figures.py
│   ├── Script_S5_make_supporting_figures.py
│   └── Script_S6_prepare_repository_manifest.py
└── dft/
    ├── inputs/
    ├── outputs/
    ├── optimized_geometries/
    ├── single_point/
    ├── cube_files/
    ├── mep_visualisations/
    └── dft_file_manifest.csv
"""
    REPO_STRUCTURE_TXT.write_text(repo_structure, encoding="utf-8")

    print("\nProject inventory completed.")
    print(f"Root: {ROOT}")
    print(f"Total files found: {len(df)}")
    print(f"\nInventory CSV:\n{INVENTORY_CSV}")
    print(f"\nTree TXT:\n{TREE_TXT}")
    print(f"\nCategory summary:\n{SUMMARY_CSV}")
    print(f"\nRecommended repository structure:\n{REPO_STRUCTURE_TXT}")
    print("\nCategory summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()