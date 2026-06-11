# -*- coding: utf-8 -*-
"""
Extract Mulliken and Loewdin atomic charges from ORCA SP outputs
for the three added FM2 DFT anchors.

Targets:
- CMR_GOLD_090
- CMR_GOLD_020
- CMR_GOLD_092

Outputs:
Per molecule:
- dft/molecules/<ID>/extracted_data/<ID>_charges_mulliken_loewdin.csv
- dft/molecules/<ID>/extracted_data/<ID>_heavy_atom_charges.csv

Combined:
- data/processed/Dataset_S33_FM2_DFT_Mulliken_Loewdin_charges.csv
- data/processed/Dataset_S33_FM2_DFT_heavy_atom_charges.csv
- data/processed/Dataset_S33_FM2_DFT_charge_extraction_report.txt
"""

from pathlib import Path
import re
import pandas as pd


ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DFT_ROOT = ROOT / "dft" / "molecules"
DATA = ROOT / "data" / "processed"

TARGET_IDS = ["CMR_GOLD_090", "CMR_GOLD_020", "CMR_GOLD_092"]


def read_lines(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return path.read_text(errors="replace").splitlines()


def parse_xyz_elements(xyz_path: Path):
    lines = read_lines(xyz_path)
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
    return atoms


def parse_charge_block(lines, header_text):
    """
    Parse ORCA charge block such as:
    MULLIKEN ATOMIC CHARGES
       0 C :   -0.123456
       1 O :   -0.456789

    Also handles variants without colon.
    """
    start_idx = None
    for i, line in enumerate(lines):
        if header_text.upper() in line.upper():
            start_idx = i
            break

    if start_idx is None:
        return {}

    charges = {}
    started = False

    # Pattern examples:
    # 0 C : -0.123
    # 0 C -0.123
    pat = re.compile(
        r"^\s*(\d+)\s+([A-Za-z]{1,3})\s*:?\s+([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)"
    )

    for line in lines[start_idx + 1:]:
        m = pat.search(line)
        if m:
            started = True
            idx = int(m.group(1))
            elem = m.group(2)
            charge = float(m.group(3))
            charges[idx] = {"element": elem, "charge": charge}
            continue

        # Skip separators before the block begins.
        if not started:
            continue

        # Once rows started, stop at blank/separator/new section.
        stripped = line.strip()
        if stripped == "":
            break
        if set(stripped) <= {"-", "="}:
            continue
        if "SUM OF" in stripped.upper():
            break
        if "LOEWDIN" in stripped.upper() or "MULLIKEN" in stripped.upper():
            break

    return charges


def parse_charges_for_molecule(cid):
    mol_dir = DFT_ROOT / cid
    sp_out = mol_dir / "output" / f"{cid}_sp.out"
    opt_xyz = mol_dir / "geometry" / f"{cid}_opt.xyz"
    extracted = mol_dir / "extracted_data"
    extracted.mkdir(parents=True, exist_ok=True)

    lines = read_lines(sp_out)

    terminated = any("ORCA TERMINATED NORMALLY" in line for line in lines)
    if not terminated:
        raise RuntimeError(f"{cid}: SP output normal bitmemiş görünüyor.")

    elements_xyz = parse_xyz_elements(opt_xyz)

    mulliken = parse_charge_block(lines, "MULLIKEN ATOMIC CHARGES")
    loewdin = parse_charge_block(lines, "LOEWDIN ATOMIC CHARGES")

    if not mulliken:
        raise RuntimeError(f"{cid}: Mulliken charge bloğu bulunamadı.")
    if not loewdin:
        raise RuntimeError(f"{cid}: Loewdin charge bloğu bulunamadı.")

    n_atoms = max(len(elements_xyz), max(mulliken.keys()) + 1, max(loewdin.keys()) + 1)

    rows = []
    for idx in range(n_atoms):
        elem = None
        if idx < len(elements_xyz):
            elem = elements_xyz[idx]
        elif idx in mulliken:
            elem = mulliken[idx]["element"]
        elif idx in loewdin:
            elem = loewdin[idx]["element"]

        rows.append({
            "Compound_ID": cid,
            "atom_index_0based": idx,
            "atom_index_1based": idx + 1,
            "element": elem,
            "Mulliken_charge": mulliken.get(idx, {}).get("charge", None),
            "Loewdin_charge": loewdin.get(idx, {}).get("charge", None),
            "is_heavy_atom": elem != "H",
            "sp_output": str(sp_out),
            "opt_xyz": str(opt_xyz),
        })

    df = pd.DataFrame(rows)

    all_csv = extracted / f"{cid}_charges_mulliken_loewdin.csv"
    heavy_csv = extracted / f"{cid}_heavy_atom_charges.csv"

    df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    df[df["is_heavy_atom"]].to_csv(heavy_csv, index=False, encoding="utf-8-sig")

    return df, all_csv, heavy_csv


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    report_lines = []

    for cid in TARGET_IDS:
        df, all_csv, heavy_csv = parse_charges_for_molecule(cid)
        all_dfs.append(df)

        report_lines.append(f"{cid}")
        report_lines.append(f"  atoms total: {len(df)}")
        report_lines.append(f"  heavy atoms: {int(df['is_heavy_atom'].sum())}")
        report_lines.append(f"  all-charge CSV: {all_csv}")
        report_lines.append(f"  heavy-atom CSV: {heavy_csv}")
        report_lines.append("")

    combined = pd.concat(all_dfs, ignore_index=True)

    out_all = DATA / "Dataset_S33_FM2_DFT_Mulliken_Loewdin_charges.csv"
    out_heavy = DATA / "Dataset_S33_FM2_DFT_heavy_atom_charges.csv"
    out_report = DATA / "Dataset_S33_FM2_DFT_charge_extraction_report.txt"

    combined.to_csv(out_all, index=False, encoding="utf-8-sig")
    combined[combined["is_heavy_atom"]].to_csv(out_heavy, index=False, encoding="utf-8-sig")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("FM2 DFT charge extraction report\n")
        f.write("=" * 70 + "\n\n")
        f.write("\n".join(report_lines))
        f.write("\nCombined outputs:\n")
        f.write(str(out_all) + "\n")
        f.write(str(out_heavy) + "\n")

    print("\nFM2 Mulliken/Löwdin charge extraction tamamlandı.")
    print(f"Combined all atoms : {out_all}")
    print(f"Combined heavy atom: {out_heavy}")
    print(f"Report             : {out_report}")

    print("\nPer-molecule summary:")
    summary = (
        combined.groupby("Compound_ID")
        .agg(
            n_atoms=("atom_index_0based", "count"),
            n_heavy_atoms=("is_heavy_atom", "sum"),
            min_Mulliken=("Mulliken_charge", "min"),
            max_Mulliken=("Mulliken_charge", "max"),
            min_Loewdin=("Loewdin_charge", "min"),
            max_Loewdin=("Loewdin_charge", "max"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\nFirst 12 heavy atoms:")
    print(combined[combined["is_heavy_atom"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()