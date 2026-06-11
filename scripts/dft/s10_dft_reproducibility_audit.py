# -*- coding: utf-8 -*-
"""
S10 DFT/MEP reproducibility audit.

Purpose:
- Scan the final DFT folder for ORCA input/output/geometry/cube/MEP files.
- Extract ORCA version where possible.
- Check normal termination.
- Count imaginary frequencies where vibrational frequencies are present.
- Extract Mulliken and Löwdin atomic charges from selected final SP outputs.
- Create SI-ready Dataset_S14–S17 files.

This script does not move, delete, or modify original DFT files.
"""

from pathlib import Path
import re
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")
PROJECT_ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp")

# Main DFT root. Change only if your final DFT folder is elsewhere.
DFT_ROOT = PROJECT / "dft"

OUT_DIR = PROJECT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MULLIKEN = OUT_DIR / "Dataset_S14_DFT_Mulliken_charges.csv"
OUT_LOEWDIN = OUT_DIR / "Dataset_S15_DFT_Lowdin_charges.csv"
OUT_STATUS = OUT_DIR / "Dataset_S16_DFT_calculation_status.csv"
OUT_MANIFEST = OUT_DIR / "Dataset_S17_DFT_file_manifest.csv"
OUT_N_SUMMARY = OUT_DIR / "Dataset_S14b_DFT_N_charge_summary.csv"


COMPOUND_MAP = {
    "055": "CMR_GOLD_055",
    "043": "CMR_GOLD_043",
    "079": "CMR_GOLD_079",
    "058": "CMR_GOLD_058",
}

TARGET_COMPOUNDS = list(COMPOUND_MAP.values())


# ============================================================
# Helper functions
# ============================================================

def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def detect_compound(path: Path) -> str:
    text = str(path)

    m = re.search(r"CMR[_-]GOLD[_-](055|043|079|058)", text, flags=re.IGNORECASE)
    if m:
        return COMPOUND_MAP[m.group(1)]

    # Folder names may simply be 055, 043, 079, 058
    parts = [p.lower() for p in path.parts]
    for code, compound in COMPOUND_MAP.items():
        if code in parts:
            return compound
        if f"_{code}" in text.lower() or f"\\{code}\\" in text.lower():
            return compound

    return "UNKNOWN"


def classify_calc_type(path: Path) -> str:
    s = str(path).lower()
    if "opt_freq" in s or "freq" in s:
        return "opt_freq"
    if "sp_charge" in s or "single" in s or "sp" in s:
        return "single_point"
    if "mep" in s or "cube" in s:
        return "mep_or_cube"
    return "unknown"


def extract_orca_version(text: str) -> str:
    patterns = [
        r"Program Version\s+([^\n\r]+)",
        r"ORCA VERSION\s*[:=]?\s*([^\n\r]+)",
        r"Version\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: find any line containing ORCA and version
    for line in text.splitlines():
        if "ORCA" in line.upper() and "VERSION" in line.upper():
            return line.strip()

    return ""


def normal_termination(text: str) -> bool:
    return "ORCA TERMINATED NORMALLY" in text.upper()


def count_imaginary_frequencies(text: str):
    """
    Try to count negative vibrational frequencies from ORCA output.
    Returns integer or None if no frequency section is found.
    """
    if "VIBRATIONAL FREQUENCIES" not in text.upper():
        return None

    freqs = []
    in_section = False

    for line in text.splitlines():
        upper = line.upper()

        if "VIBRATIONAL FREQUENCIES" in upper:
            in_section = True
            continue

        if in_section:
            if "NORMAL MODES" in upper or "IR SPECTRUM" in upper:
                break

            # Common ORCA format: "  0:        23.45 cm**-1"
            m = re.match(r"\s*\d+\s*:\s*(-?\d+(?:\.\d+)?)", line)
            if m:
                freqs.append(float(m.group(1)))

    if not freqs:
        return None

    return int(sum(1 for f in freqs if f < 0))


def section_lines(text: str, section_title: str):
    """
    Extract lines after a section title until a blank line or next obvious section.
    """
    lines = text.splitlines()
    start = None

    title_upper = section_title.upper()

    for i, line in enumerate(lines):
        if title_upper in line.upper():
            start = i + 1
            break

    if start is None:
        return []

    out = []
    for line in lines[start:]:
        upper = line.upper().strip()

        if not line.strip() and out:
            break

        if out and (
            "SUM OF" in upper
            or "LOEWDIN" in upper
            or "MULLIKEN" in upper
            or "HIRSHFELD" in upper
            or "CARTESIAN" in upper
            or "DIPOLE" in upper
        ):
            break

        out.append(line)

    return out


def parse_atomic_charge_section(text: str, section_title: str, compound: str, source_file: Path):
    """
    Parse atomic charge sections like:
        0 C : -0.123456
        1 N :  0.234567
    """
    lines = section_lines(text, section_title)
    records = []

    for line in lines:
        m = re.match(
            r"\s*(\d+)\s+([A-Za-z]{1,2})\s*:\s*(-?\d+(?:\.\d+)?)",
            line
        )
        if not m:
            continue

        atom_index = int(m.group(1))
        element = m.group(2)
        charge = float(m.group(3))

        records.append({
            "Compound": compound,
            "Atom_index_ORCA": atom_index,
            "Element": element,
            "Charge": charge,
            "Source_file": str(source_file.relative_to(PROJECT_ROOT)),
        })

    return records


def choose_best_sp_file(out_files):
    """
    Prefer normal-terminated single-point files with Mulliken/Löwdin sections.
    """
    candidates = []

    for f in out_files:
        text = safe_read_text(f)
        calc_type = classify_calc_type(f)
        has_mulliken = "MULLIKEN ATOMIC CHARGES" in text.upper()
        has_loewdin = "LOEWDIN ATOMIC CHARGES" in text.upper()
        terminated = normal_termination(text)

        score = 0
        if calc_type == "single_point":
            score += 5
        if "charge" in str(f).lower():
            score += 3
        if terminated:
            score += 2
        if has_mulliken:
            score += 2
        if has_loewdin:
            score += 2

        candidates.append((score, f.stat().st_mtime, f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


# ============================================================
# Main scan
# ============================================================

if not DFT_ROOT.exists():
    raise FileNotFoundError(f"DFT_ROOT bulunamadı: {DFT_ROOT}")

all_files = [p for p in DFT_ROOT.rglob("*") if p.is_file()]
out_files = [p for p in all_files if p.suffix.lower() == ".out"]

manifest_rows = []
status_rows = []
mulliken_rows = []
loewdin_rows = []

# Manifest for all relevant DFT files
for f in all_files:
    compound = detect_compound(f)
    manifest_rows.append({
        "Compound": compound,
        "Relative_path": str(f.relative_to(PROJECT_ROOT)),
        "File_name": f.name,
        "Extension": f.suffix.lower(),
        "File_size_bytes": f.stat().st_size,
        "Calculation_type_guess": classify_calc_type(f),
    })

# Group out files by compound
compound_to_out = {compound: [] for compound in TARGET_COMPOUNDS}
unknown_out = []

for f in out_files:
    compound = detect_compound(f)
    if compound in compound_to_out:
        compound_to_out[compound].append(f)
    else:
        unknown_out.append(f)

for compound in TARGET_COMPOUNDS:
    files = compound_to_out.get(compound, [])

    opt_files = [f for f in files if classify_calc_type(f) == "opt_freq"]
    sp_files = [f for f in files if classify_calc_type(f) == "single_point"]

    # Status fields
    opt_status = "Not found"
    opt_imag = ""
    opt_version = ""
    opt_file = ""

    if opt_files:
        # prefer newest normal-terminated opt/freq file
        best_opt = sorted(opt_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        text_opt = safe_read_text(best_opt)
        opt_status = "Complete" if normal_termination(text_opt) else "Check"
        opt_imag_value = count_imaginary_frequencies(text_opt)
        opt_imag = "" if opt_imag_value is None else opt_imag_value
        opt_version = extract_orca_version(text_opt)
        opt_file = str(best_opt.relative_to(PROJECT_ROOT))

    best_sp = choose_best_sp_file(files)

    sp_status = "Not found"
    sp_version = ""
    sp_file = ""

    if best_sp is not None:
        text_sp = safe_read_text(best_sp)
        sp_status = "Complete" if normal_termination(text_sp) else "Check"
        sp_version = extract_orca_version(text_sp)
        sp_file = str(best_sp.relative_to(PROJECT_ROOT))

        mulliken_rows.extend(
            parse_atomic_charge_section(
                text_sp,
                "MULLIKEN ATOMIC CHARGES",
                compound,
                best_sp,
            )
        )

        loewdin_rows.extend(
            parse_atomic_charge_section(
                text_sp,
                "LOEWDIN ATOMIC CHARGES",
                compound,
                best_sp,
            )
        )

    version = sp_version or opt_version

    status_rows.append({
        "Compound": compound,
        "ORCA_version": version,
        "Opt_Freq_status": opt_status,
        "Imaginary_frequencies": opt_imag,
        "SP_status": sp_status,
        "Selected_opt_freq_output": opt_file,
        "Selected_SP_output": sp_file,
        "Number_of_output_files_found": len(files),
    })


# ============================================================
# Write outputs
# ============================================================

manifest_df = pd.DataFrame(manifest_rows)
status_df = pd.DataFrame(status_rows)

mulliken_df = pd.DataFrame(mulliken_rows)
loewdin_df = pd.DataFrame(loewdin_rows)

manifest_df.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")
status_df.to_csv(OUT_STATUS, index=False, encoding="utf-8-sig")

if not mulliken_df.empty:
    mulliken_df.to_csv(OUT_MULLIKEN, index=False, encoding="utf-8-sig")
else:
    pd.DataFrame(columns=["Compound", "Atom_index_ORCA", "Element", "Charge", "Source_file"]).to_csv(
        OUT_MULLIKEN, index=False, encoding="utf-8-sig"
    )

if not loewdin_df.empty:
    loewdin_df.to_csv(OUT_LOEWDIN, index=False, encoding="utf-8-sig")
else:
    pd.DataFrame(columns=["Compound", "Atom_index_ORCA", "Element", "Charge", "Source_file"]).to_csv(
        OUT_LOEWDIN, index=False, encoding="utf-8-sig"
    )

# N-only charge summary for Table S12
if not mulliken_df.empty or not loewdin_df.empty:
    m_n = mulliken_df[mulliken_df["Element"].str.upper() == "N"].copy() if not mulliken_df.empty else pd.DataFrame()
    l_n = loewdin_df[loewdin_df["Element"].str.upper() == "N"].copy() if not loewdin_df.empty else pd.DataFrame()

    if not m_n.empty:
        m_n = m_n.rename(columns={"Charge": "Mulliken_charge"})
    if not l_n.empty:
        l_n = l_n.rename(columns={"Charge": "Lowdin_charge"})

    if not m_n.empty and not l_n.empty:
        n_summary = pd.merge(
            m_n[["Compound", "Atom_index_ORCA", "Element", "Mulliken_charge", "Source_file"]],
            l_n[["Compound", "Atom_index_ORCA", "Lowdin_charge"]],
            on=["Compound", "Atom_index_ORCA"],
            how="outer",
        )
    elif not m_n.empty:
        n_summary = m_n[["Compound", "Atom_index_ORCA", "Element", "Mulliken_charge", "Source_file"]].copy()
        n_summary["Lowdin_charge"] = ""
    elif not l_n.empty:
        n_summary = l_n[["Compound", "Atom_index_ORCA", "Element", "Lowdin_charge", "Source_file"]].copy()
        n_summary["Mulliken_charge"] = ""

    n_summary = n_summary.sort_values(["Compound", "Atom_index_ORCA"])
    n_summary.to_csv(OUT_N_SUMMARY, index=False, encoding="utf-8-sig")
else:
    pd.DataFrame(columns=[
        "Compound", "Atom_index_ORCA", "Element",
        "Mulliken_charge", "Lowdin_charge", "Source_file"
    ]).to_csv(OUT_N_SUMMARY, index=False, encoding="utf-8-sig")


# ============================================================
# Console summary
# ============================================================

print("\nS10 DFT/MEP audit completed.")
print(f"DFT root: {DFT_ROOT}")
print(f"Total DFT files scanned: {len(all_files)}")
print(f"Total ORCA .out files scanned: {len(out_files)}")
print("")
print("Output files:")
print(f"  {OUT_MULLIKEN}")
print(f"  {OUT_LOEWDIN}")
print(f"  {OUT_STATUS}")
print(f"  {OUT_MANIFEST}")
print(f"  {OUT_N_SUMMARY}")
print("")
print("Calculation status:")
print(status_df.to_string(index=False))

if not mulliken_df.empty:
    print("\nMulliken N charge preview:")
    print(mulliken_df[mulliken_df["Element"].str.upper() == "N"].head(20).to_string(index=False))
else:
    print("\nNo Mulliken charges parsed.")

if not loewdin_df.empty:
    print("\nLöwdin N charge preview:")
    print(loewdin_df[loewdin_df["Element"].str.upper() == "N"].head(20).to_string(index=False))
else:
    print("\nNo Löwdin charges parsed.")