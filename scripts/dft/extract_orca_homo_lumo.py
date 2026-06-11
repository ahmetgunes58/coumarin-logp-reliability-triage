# -*- coding: utf-8 -*-
"""
Extract final energy, HOMO, LUMO, and HOMO-LUMO gap from an ORCA output file.

Usage:
    python extract_orca_homo_lumo.py dft\molecules\CMR_GOLD_090\output\CMR_GOLD_090_sp.out
"""

from pathlib import Path
import sys
import re
import pandas as pd


def parse_orca_output(out_path: Path):
    text = out_path.read_text(errors="replace")
    lines = text.splitlines()

    terminated_normally = "ORCA TERMINATED NORMALLY" in text

    # Final single point energy
    energy_matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    final_energy_eh = float(energy_matches[-1]) if energy_matches else None

    # Find last ORBITAL ENERGIES block
    start_indices = [i for i, line in enumerate(lines) if "ORBITAL ENERGIES" in line]
    if not start_indices:
        raise RuntimeError("ORBITAL ENERGIES bloğu bulunamadı.")

    start = start_indices[-1]
    block = []

    for line in lines[start + 1:]:
        if "MOLECULAR ORBITALS" in line:
            break
        if "MULLIKEN" in line.upper():
            break
        if "LOEWDIN" in line.upper():
            break
        if "DIPOLE MOMENT" in line.upper():
            break
        block.append(line)

    rows = []

    for line in block:
        parts = line.split()

        # ORCA orbital table usually:
        # NO   OCC          E(Eh)            E(eV)
        # 123  2.0000      -0.250000        -6.8027
        if len(parts) >= 4:
            try:
                orb_no = int(parts[0])
                occ = float(parts[1])
                e_eh = float(parts[2])
                e_ev = float(parts[3])
                rows.append({
                    "orbital_no": orb_no,
                    "occupation": occ,
                    "energy_Eh": e_eh,
                    "energy_eV": e_ev,
                    "raw_line": line,
                })
            except Exception:
                pass

    if not rows:
        raise RuntimeError("ORBITAL ENERGIES bloğundan orbital satırı okunamadı.")

    occupied = [r for r in rows if r["occupation"] > 0.0]
    virtual = [r for r in rows if r["occupation"] == 0.0]

    if not occupied:
        raise RuntimeError("Occupied orbital bulunamadı.")
    if not virtual:
        raise RuntimeError("Virtual orbital bulunamadı.")

    homo = occupied[-1]
    lumo = virtual[0]
    gap_ev = lumo["energy_eV"] - homo["energy_eV"]

    return {
        "output_file": str(out_path),
        "terminated_normally": terminated_normally,
        "final_energy_Eh": final_energy_eh,
        "HOMO_orbital": homo["orbital_no"],
        "HOMO_eV": homo["energy_eV"],
        "LUMO_orbital": lumo["orbital_no"],
        "LUMO_eV": lumo["energy_eV"],
        "Gap_eV": gap_ev,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_orca_homo_lumo.py path_to_orca_output.out")
        sys.exit(1)

    out_path = Path(sys.argv[1])

    if not out_path.exists():
        raise FileNotFoundError(f"Output dosyası bulunamadı: {out_path}")

    result = parse_orca_output(out_path)

    print("\nORCA frontier-orbital extraction")
    print("=" * 50)
    for k, v in result.items():
        print(f"{k}: {v}")

    # Try to save into molecule extracted_data folder if path follows dft/molecules/ID/output/file.out
    try:
        mol_dir = out_path.parent.parent
        extracted_dir = mol_dir / "extracted_data"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        cid = mol_dir.name
        out_csv = extracted_dir / f"{cid}_frontier_orbitals.csv"
        pd.DataFrame([result]).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved CSV: {out_csv}")
    except Exception as e:
        print(f"\nCSV kaydı atlandı: {e}")


if __name__ == "__main__":
    main()