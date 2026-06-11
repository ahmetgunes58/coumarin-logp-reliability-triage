# -*- coding: utf-8 -*-
"""
Scan C:\\orca_tests\\DFT_FINAL_CLEAN for DFT panel ORCA output files.

Corrected version:
- detects compound code from file/folder names containing CMR_GOLD_055, 043, 079, 058
- avoids false matches from dates such as 20260430
- reads full .out text for charge-section detection
- writes Dataset_S17a_orca_tests_output_candidates.csv

This script does not copy, move, delete, or modify files.
"""

from pathlib import Path
import re
import pandas as pd


DFT_ROOT = Path(r"C:\orca_tests\DFT_FINAL_CLEAN")
PROJECT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")

OUT_REPORT = PROJECT / "data" / "processed" / "Dataset_S17a_orca_tests_output_candidates.csv"
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

TARGETS = ["055", "043", "079", "058"]


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def detect_code(path: Path) -> str:
    """
    Detect compound code only from reliable CMR_GOLD-style names,
    not from arbitrary digits in folder names or dates.
    """
    s = str(path)

    # Prefer full CMR_GOLD_XXX / CMR-GOLD-XXX patterns
    m = re.search(r"CMR[_-]GOLD[_-](055|043|079|058)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Fallback: exact filename contains _XXX_ or _XXX.
    name = path.name
    m = re.search(r"(?:^|[_-])(055|043|079|058)(?:[_\.-]|$)", name)
    if m:
        return m.group(1)

    return ""


def detect_calc_type(path: Path) -> str:
    s = str(path).lower()
    name = path.name.lower()

    if "sp_charge" in s or "sp_charge" in name or "charge" in name:
        return "single_point"

    if "opt_freq" in s or "correct_opt" in name or "freq" in s:
        return "opt_freq"

    return "unknown"


def detect_orca_version(text: str) -> str:
    patterns = [
        r"Program Version\s+([^\n\r]+)",
        r"ORCA VERSION\s*[:=]?\s*([^\n\r]+)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    for line in text.splitlines():
        if "ORCA" in line.upper() and "VERSION" in line.upper():
            return line.strip()

    return ""


def has_normal_termination(text: str) -> bool:
    return "ORCA TERMINATED NORMALLY" in text.upper()


def count_imaginary_frequencies(text: str):
    if "VIBRATIONAL FREQUENCIES" not in text.upper():
        return ""

    freqs = []
    in_section = False

    for line in text.splitlines():
        up = line.upper()

        if "VIBRATIONAL FREQUENCIES" in up:
            in_section = True
            continue

        if in_section and ("NORMAL MODES" in up or "IR SPECTRUM" in up):
            break

        if in_section:
            m = re.match(r"\s*\d+\s*:\s*(-?\d+(?:\.\d+)?)", line)
            if m:
                freqs.append(float(m.group(1)))

    if not freqs:
        return ""

    return int(sum(1 for f in freqs if f < 0))


def has_charge_sections(text: str) -> tuple[bool, bool]:
    up = text.upper()

    has_mulliken = (
        "MULLIKEN ATOMIC CHARGES" in up
        or "MULLIKEN CHARGES" in up
    )

    has_lowdin = (
        "LOEWDIN ATOMIC CHARGES" in up
        or "LÖWDIN ATOMIC CHARGES" in up
        or "LOWDIN ATOMIC CHARGES" in up
        or "LOEWDIN CHARGES" in up
        or "LOWDIN CHARGES" in up
    )

    return has_mulliken, has_lowdin


def main():
    if not DFT_ROOT.exists():
        raise FileNotFoundError(f"DFT root not found: {DFT_ROOT}")

    all_out = list(DFT_ROOT.rglob("*.out"))

    rows = []

    for p in all_out:
        code = detect_code(p)
        if code not in TARGETS:
            continue

        text = safe_read_text(p)
        mulliken, lowdin = has_charge_sections(text)

        rows.append({
            "Code": code,
            "Compound": f"CMR_GOLD_{code}",
            "Candidate_calc_type": detect_calc_type(p),
            "File_name": p.name,
            "Full_path": str(p),
            "Size_MB": round(p.stat().st_size / (1024 * 1024), 3),
            "Modified_time": pd.to_datetime(p.stat().st_mtime, unit="s"),
            "ORCA_version": detect_orca_version(text),
            "Normal_termination": has_normal_termination(text),
            "Imaginary_frequencies": count_imaginary_frequencies(text),
            "Has_Mulliken_charges": mulliken,
            "Has_Lowdin_charges": lowdin,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No candidate .out files found.")
        return

    df = df.sort_values(
        ["Compound", "Candidate_calc_type", "Modified_time"],
        ascending=[True, True, False]
    )

    df.to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")

    print("\nCorrected candidate ORCA output scan completed.")
    print(f"DFT root: {DFT_ROOT}")
    print(f"Candidate .out files found: {len(df)}")
    print(f"Report: {OUT_REPORT}")

    print("\nSummary by compound/type:")
    print(df.groupby(["Compound", "Candidate_calc_type"]).size().to_string())

    print("\nCandidate files:")
    print(df[[
        "Compound",
        "Candidate_calc_type",
        "File_name",
        "Normal_termination",
        "Imaginary_frequencies",
        "Has_Mulliken_charges",
        "Has_Lowdin_charges",
        "ORCA_version",
        "Size_MB",
        "Full_path",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()