# -*- coding: utf-8 -*-
"""
Import existing finalized DFT files for older DFT anchors into the current repository layout.

This script normalizes filenames into:
dft/molecules/<ID>/output/<ID>_optfreq.out
dft/molecules/<ID>/output/<ID>_sp.out
dft/molecules/<ID>/geometry/<ID>_opt.xyz

It does not rerun ORCA.
"""

from pathlib import Path
import shutil
import pandas as pd


ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DFT_ROOT = ROOT / "dft" / "molecules"
DATA = ROOT / "data" / "processed"

ANCHORS = {
    "CMR_GOLD_043": {
        "opt_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\01_opt_freq_correct\CMR_GOLD_043_correct_opt_pal8.out"),
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\01_opt_freq\CMR_GOLD_043_correct_opt_pal8.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot-v2-final\molecules\CMR_GOLD_043\01_opt_freq\CMR_GOLD_043_correct_opt_pal8.out"),
        ],
        "sp_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\02_sp_charge_correct\CMR_GOLD_043_sp_charge_clean.out"),
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\02_sp_charge\CMR_GOLD_043_sp_charge_clean.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot-v2-final\molecules\CMR_GOLD_043\02_sp_charge\CMR_GOLD_043_sp_charge_clean.out"),
        ],
        "opt_xyz": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\01_opt_freq_correct\CMR_GOLD_043_correct_opt_pal8.xyz"),
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\01_opt_freq\CMR_GOLD_043_correct_opt_pal8.xyz"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot-v2-final\molecules\CMR_GOLD_043\01_opt_freq\CMR_GOLD_043_correct_opt_pal8.xyz"),
        ],
    },

    "CMR_GOLD_055": {
        "opt_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\055\01_opt_freq\CMR_GOLD_055_correct_opt_pal8_clean.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_055\01_opt_freq\CMR_GOLD_055_correct_opt_pal8_clean.out"),
        ],
        "sp_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\055\02_sp_charge\CMR_GOLD_055_sp_charge_clean.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_055\02_sp_charge\CMR_GOLD_055_sp_charge_clean.out"),
        ],
        "opt_xyz": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\055\01_opt_freq\CMR_GOLD_055_correct_opt_pal8_clean.xyz"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_055\01_opt_freq\CMR_GOLD_055_correct_opt_pal8_clean.xyz"),
        ],
    },

    "CMR_GOLD_058": {
        "opt_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\058\01_opt_freq\CMR_GOLD_058_correct_opt_pal8.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_058\01_opt_freq\CMR_GOLD_058_correct_opt_pal8.out"),
        ],
        "sp_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\058\02_sp_charge\CMR_GOLD_058_sp_charge_clean.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_058\02_sp_charge\CMR_GOLD_058_sp_charge_clean.out"),
        ],
        "opt_xyz": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\058\01_opt_freq\CMR_GOLD_058_correct_opt_pal8.xyz"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_058\01_opt_freq\CMR_GOLD_058_correct_opt_pal8.xyz"),
        ],
    },

    "CMR_GOLD_079": {
        "opt_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\079\01_opt_freq\CMR_GOLD_079_correct_opt_pal8.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_079\01_opt_freq\CMR_GOLD_079_correct_opt_pal8.out"),
        ],
        "sp_out": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\079\02_sp_charge\CMR_GOLD_079_sp_charge_clean.out"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_079\02_sp_charge\CMR_GOLD_079_sp_charge_clean.out"),
        ],
        "opt_xyz": [
            Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\079\01_opt_freq\CMR_GOLD_079_correct_opt_pal8.xyz"),
            Path(r"C:\orca_tests\DFT_FINAL_CLEAN\dft-panel-prior-snapshot\molecules\CMR_GOLD_079\01_opt_freq\CMR_GOLD_079_correct_opt_pal8.xyz"),
        ],
    },
}


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    rows = []

    for cid, files in ANCHORS.items():
        mol_dir = DFT_ROOT / cid
        out_dir = mol_dir / "output"
        geom_dir = mol_dir / "geometry"
        extracted_dir = mol_dir / "extracted_data"

        out_dir.mkdir(parents=True, exist_ok=True)
        geom_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        opt_src = first_existing(files["opt_out"])
        sp_src = first_existing(files["sp_out"])
        xyz_src = first_existing(files["opt_xyz"])

        opt_dst = out_dir / f"{cid}_optfreq.out"
        sp_dst = out_dir / f"{cid}_sp.out"
        xyz_dst = geom_dir / f"{cid}_opt.xyz"

        status = {
            "Compound_ID": cid,
            "opt_src": str(opt_src) if opt_src else "",
            "sp_src": str(sp_src) if sp_src else "",
            "xyz_src": str(xyz_src) if xyz_src else "",
            "opt_imported": False,
            "sp_imported": False,
            "xyz_imported": False,
            "opt_dst": str(opt_dst),
            "sp_dst": str(sp_dst),
            "xyz_dst": str(xyz_dst),
        }

        if opt_src:
            copy_file(opt_src, opt_dst)
            status["opt_imported"] = True

        if sp_src:
            copy_file(sp_src, sp_dst)
            status["sp_imported"] = True

        if xyz_src:
            copy_file(xyz_src, xyz_dst)
            status["xyz_imported"] = True

        rows.append(status)

    report = pd.DataFrame(rows)
    out_csv = DATA / "Dataset_S35_imported_existing_DFT_anchor_files.csv"
    DATA.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\nExisting DFT anchor import completed.")
    print(f"Report: {out_csv}")
    print(report.to_string(index=False))

    missing = report[
        (~report["opt_imported"]) |
        (~report["sp_imported"]) |
        (~report["xyz_imported"])
    ]

    if len(missing):
        print("\nWARNING: Some files were not imported:")
        print(missing.to_string(index=False))
    else:
        print("\nAll requested old-anchor files were imported successfully.")


if __name__ == "__main__":
    main()