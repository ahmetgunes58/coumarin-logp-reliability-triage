# -*- coding: utf-8 -*-
"""
Build artifact manifest for the completed DFT upgrade.

This manifest records:
- Data outputs Dataset_S29–S40
- DFT workflow scripts
- 10-compound DFT panel molecular outputs
- Added FM2 MEP/cube outputs
- Missing-file status

Outputs:
- data/processed/Dataset_S41_DFT_upgrade_artifact_manifest.csv
- data/processed/Dataset_S41_DFT_upgrade_artifact_manifest_report.txt
"""

from pathlib import Path
import pandas as pd


ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DATA = ROOT / "data" / "processed"
DFT = ROOT / "dft" / "molecules"

PANEL_IDS = [
    "CMR_GOLD_016",
    "CMR_GOLD_029",
    "CMR_GOLD_044",
    "CMR_GOLD_043",
    "CMR_GOLD_055",
    "CMR_GOLD_058",
    "CMR_GOLD_079",
    "CMR_GOLD_090",
    "CMR_GOLD_020",
    "CMR_GOLD_092",
]

FM2_NEW_IDS = ["CMR_GOLD_090", "CMR_GOLD_020", "CMR_GOLD_092"]

DATASET_FILES = [
    "Dataset_S29_FM2_DFT_selected_anchors.csv",
    "Dataset_S29_FM2_DFT_input_file_manifest.csv",
    "Dataset_S32_FM2_DFT_anchor_results.csv",
    "Dataset_S32_FM2_DFT_anchor_status_report.txt",
    "Dataset_S33_FM2_DFT_Mulliken_Loewdin_charges.csv",
    "Dataset_S33_FM2_DFT_heavy_atom_charges.csv",
    "Dataset_S33_FM2_DFT_charge_extraction_report.txt",
    "Dataset_S34_DFT_panel_10_summary.csv",
    "Dataset_S34_DFT_panel_10_status_report.txt",
    "Dataset_S35_imported_existing_DFT_anchor_files.csv",
    "Dataset_S36_imported_DFT_anchor_Mulliken_Loewdin_charges.csv",
    "Dataset_S36_imported_DFT_anchor_heavy_atom_charges.csv",
    "Dataset_S36_imported_DFT_anchor_charge_extraction_report.txt",
    "Dataset_S37_DFT_panel_10_Mulliken_Loewdin_charges.csv",
    "Dataset_S37_DFT_panel_10_heavy_atom_charges.csv",
    "Dataset_S38_DFT_panel_10_charge_summary.csv",
    "Dataset_S39_DFT_panel_10_frontier_charge_summary.csv",
    "Dataset_S39_DFT_panel_10_final_report.txt",
    "Dataset_S40_FM2_MEP_cube_validation_summary.csv",
    "Dataset_S40_FM2_MEP_cube_generation_report.txt",
]

SCRIPT_FILES = [
    "prepare_FM2_DFT_inputs.py",
    "extract_orca_homo_lumo.py",
    "summarize_FM2_DFT_results.py",
    "extract_FM2_charges.py",
    "import_existing_DFT_anchors.py",
    "build_DFT_panel_10_summary.py",
    "extract_missing_DFT_panel_charges.py",
    "build_DFT_panel_10_final_tables.py",
    "fix_FM2_orcaplot_paths.py",
    "generate_FM2_MEP_cubes_orcaplot.py",
    "build_DFT_upgrade_artifact_manifest.py",
]


def file_record(path: Path, category: str, description: str, compound_id: str = ""):
    exists = path.exists()
    size_bytes = path.stat().st_size if exists and path.is_file() else None

    try:
        rel_path = str(path.relative_to(ROOT))
    except Exception:
        rel_path = str(path)

    return {
        "Category": category,
        "Compound_ID": compound_id,
        "Description": description,
        "Relative_path": rel_path,
        "Absolute_path": str(path),
        "Exists": exists,
        "Size_bytes": size_bytes,
    }


def main():
    rows = []

    # Dataset outputs
    for fname in DATASET_FILES:
        rows.append(
            file_record(
                DATA / fname,
                "processed_dataset_or_report",
                "DFT upgrade processed output / report",
            )
        )

    # Scripts
    for fname in SCRIPT_FILES:
        rows.append(
            file_record(
                ROOT / fname,
                "script",
                "Script used in DFT upgrade / validation workflow",
            )
        )

    # Required per-molecule DFT files for 10-compound panel
    for cid in PANEL_IDS:
        mol_dir = DFT / cid

        required_files = [
            (mol_dir / "output" / f"{cid}_optfreq.out", "ORCA Opt/Freq output"),
            (mol_dir / "output" / f"{cid}_sp.out", "ORCA single-point output"),
            (mol_dir / "geometry" / f"{cid}_opt.xyz", "Optimized geometry XYZ"),
            (mol_dir / "extracted_data" / f"{cid}_charges_mulliken_loewdin.csv", "Mulliken/Löwdin all-atom charge table"),
            (mol_dir / "extracted_data" / f"{cid}_heavy_atom_charges.csv", "Mulliken/Löwdin heavy-atom charge table"),
        ]

        for path, desc in required_files:
            rows.append(file_record(path, "dft_panel_required_file", desc, cid))

    # Added FM2 cube / MEP package files
    for cid in FM2_NEW_IDS:
        mol_dir = DFT / cid

        fm2_files = [
            (mol_dir / "cube" / f"{cid}_density.cube", "Final electron-density cube"),
            (mol_dir / "cube" / f"{cid}_esp.cube", "Final electrostatic-potential cube"),
            (mol_dir / "cube" / f"{cid}_sp.eldens.cube", "ORCA-generated electron-density cube"),
            (mol_dir / "cube" / "input" / f"{cid}_sp.scfp.esp.cube", "ORCA-generated ESP cube in recursive input subfolder"),
            (mol_dir / "extracted_data" / f"{cid}_cube_validation_report.txt", "Cube validation report TXT"),
            (mol_dir / "extracted_data" / f"{cid}_cube_validation_report.csv", "Cube validation report CSV"),
            (mol_dir / "extracted_data" / f"{cid}_orcaplot_density.log", "orca_plot electron-density log"),
            (mol_dir / "extracted_data" / f"{cid}_orcaplot_esp.log", "orca_plot ESP log"),
        ]

        for path, desc in fm2_files:
            rows.append(file_record(path, "fm2_mep_cube_package", desc, cid))

    manifest = pd.DataFrame(rows)

    out_csv = DATA / "Dataset_S41_DFT_upgrade_artifact_manifest.csv"
    out_report = DATA / "Dataset_S41_DFT_upgrade_artifact_manifest_report.txt"

    DATA.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(out_csv, index=False, encoding="utf-8-sig")

    missing = manifest[manifest["Exists"] != True].copy()

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("DFT upgrade artifact manifest report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total manifest entries: {len(manifest)}\n")
        f.write(f"Existing entries      : {int(manifest['Exists'].sum())}\n")
        f.write(f"Missing entries       : {len(missing)}\n\n")

        f.write("Category counts:\n")
        f.write(manifest.groupby("Category")["Relative_path"].count().to_string())
        f.write("\n\n")

        if len(missing):
            f.write("Missing entries:\n")
            f.write(missing[["Category", "Compound_ID", "Description", "Relative_path"]].to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No missing entries detected.\n\n")

        f.write("Full manifest:\n")
        f.write(manifest.to_string(index=False))
        f.write("\n")

    print("\nDFT upgrade artifact manifest generated.")
    print(f"CSV   : {out_csv}")
    print(f"Report: {out_report}")

    print("\nSummary:")
    print(f"Total entries : {len(manifest)}")
    print(f"Existing      : {int(manifest['Exists'].sum())}")
    print(f"Missing       : {len(missing)}")

    print("\nCategory counts:")
    print(manifest.groupby("Category")["Relative_path"].count().to_string())

    if len(missing):
        print("\nWARNING: Missing entries:")
        print(missing[["Category", "Compound_ID", "Description", "Relative_path"]].to_string(index=False))
    else:
        print("\nAll manifest entries exist.")


if __name__ == "__main__":
    main()