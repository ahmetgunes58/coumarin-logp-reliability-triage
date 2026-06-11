# -*- coding: utf-8 -*-
"""
Scan for existing MEP/ESP visualisation and cube files for Figure S5.

Targets:
    CMR_GOLD_055
    CMR_GOLD_079

This script does not move, delete, or modify files.
It writes a candidate report to:
    coumarin-logp/data/processed/Dataset_S17c_Figure_S5_MEP_candidates.csv
"""

from pathlib import Path
import pandas as pd


PROJECT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")
PROJECT_ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp")
ORCA_TESTS_ROOT = Path(r"C:\orca_tests\DFT_FINAL_CLEAN")

SEARCH_ROOTS = [
    PROJECT,
    PROJECT_ROOT,
    ORCA_TESTS_ROOT,
]

OUT_CSV = PROJECT / "data" / "processed" / "Dataset_S17c_Figure_S5_MEP_candidates.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TARGETS = ["CMR_GOLD_055", "CMR_GOLD_079"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svg", ".pdf"}
CUBE_EXTS = {".cube", ".cub"}
STRUCT_EXTS = {".xyz", ".molden", ".molden.input", ".gbw"}
TEXT_EXTS = {".inp", ".out", ".txt", ".log"}

KEYWORDS = [
    "mep",
    "esp",
    "electrostatic",
    "potential",
    "density",
    "rho",
    "surface",
    "vmd",
    "render",
]


def detect_target(path: Path):
    s = str(path).lower()
    for target in TARGETS:
        if target.lower() in s:
            return target
    return ""


def classify_file(path: Path):
    ext = path.suffix.lower()
    s = str(path).lower()

    if ext in IMAGE_EXTS:
        if any(k in s for k in KEYWORDS):
            return "MEP_or_visual_image"
        return "image_candidate"

    if ext in CUBE_EXTS:
        if "esp" in s or "potential" in s:
            return "ESP_cube"
        if "density" in s or "rho" in s:
            return "density_cube"
        return "cube_candidate"

    if ext in STRUCT_EXTS:
        return "structure_or_wavefunction"

    if ext in TEXT_EXTS:
        return "orca_text_or_input"

    return "other"


def score_file(path: Path):
    s = str(path).lower()
    score = 0

    if path.suffix.lower() in IMAGE_EXTS:
        score += 10
    if path.suffix.lower() in CUBE_EXTS:
        score += 7

    for k in KEYWORDS:
        if k in s:
            score += 2

    if "mep" in s:
        score += 5
    if "esp" in s:
        score += 4
    if "final" in s:
        score += 2

    return score


def main():
    rows = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            target = detect_target(path)
            if not target:
                continue

            ext = path.suffix.lower()
            if ext not in IMAGE_EXTS | CUBE_EXTS | STRUCT_EXTS | TEXT_EXTS:
                continue

            resolved = str(path.resolve()).lower()
            if resolved in seen:
                continue
            seen.add(resolved)

            rows.append({
                "Target": target,
                "File_name": path.name,
                "Full_path": str(path),
                "Extension": ext,
                "Class": classify_file(path),
                "Score": score_file(path),
                "Size_MB": round(path.stat().st_size / (1024 * 1024), 3),
                "Modified_time": pd.to_datetime(path.stat().st_mtime, unit="s"),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No Figure S5 MEP/ESP candidates found.")
        return

    df = df.sort_values(["Target", "Score", "Modified_time"], ascending=[True, False, False])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\nFigure S5 MEP candidate scan completed.")
    print(f"Report: {OUT_CSV}")
    print("\nCandidate summary:")
    print(df.groupby(["Target", "Class"]).size().to_string())
    print("\nTop candidates:")
    print(df[[
        "Target",
        "Class",
        "Score",
        "File_name",
        "Size_MB",
        "Full_path",
    ]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()