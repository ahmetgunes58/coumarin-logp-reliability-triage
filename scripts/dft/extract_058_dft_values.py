# -*- coding: utf-8 -*-
"""
Extract HOMO, LUMO, gap, dipole, and N-atom charges
from corrected CMR_GOLD_058 ORCA output.
"""

import re
from pathlib import Path

OUT_FILE = "CMR_GOLD_058_sp_charge_clean.out"
XYZ_FILE = "CMR_GOLD_058_correct_opt_pal8.xyz"

def read_xyz_elements(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    n = int(lines[0].strip())
    elems = []
    for line in lines[2:2+n]:
        parts = line.split()
        elems.append(parts[0])
    return elems

def extract_dipole(text):
    m = re.search(r"Magnitude\s+\(Debye\)\s*:\s*([-+]?\d+\.\d+)", text)
    return float(m.group(1)) if m else None

def extract_final_energy(text):
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", text)
    return float(m.group(1)) if m else None

def extract_orbitals(text):
    marker = "ORBITAL ENERGIES"
    start = text.find(marker)
    if start == -1:
        return []

    end_markers = [
        "MULLIKEN",
        "LOEWDIN",
        "Total SCF time",
        "FINAL SINGLE POINT ENERGY",
    ]

    end = len(text)
    for em in end_markers:
        pos = text.find(em, start + len(marker))
        if pos != -1:
            end = min(end, pos)

    section = text[start:end]
    orbitals = []

    # Typical ORCA line:
    #   132   2.0000    -0.123456     -3.3599
    for line in section.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                no = int(parts[0])
                occ = float(parts[1])
                e_hartree = float(parts[2])
                e_ev = float(parts[3])
                orbitals.append((no, occ, e_hartree, e_ev))
            except Exception:
                pass

    return orbitals

def extract_atomic_charges(text, heading):
    start = text.find(heading)
    if start == -1:
        return {}

    # stop at next major section
    possible_ends = [
        "Sum of atomic charges",
        "LOEWDIN ATOMIC CHARGES",
        "MULLIKEN REDUCED",
        "LOEWDIN REDUCED",
        "Mayer",
        "ORCA PROPERTY",
        "FINAL SINGLE POINT ENERGY",
    ]

    end = len(text)
    for em in possible_ends:
        pos = text.find(em, start + len(heading))
        if pos != -1:
            end = min(end, pos)

    section = text[start:end]
    charges = {}

    # Common ORCA atomic charge line:
    #   0 C :   -0.123456
    #   1 N :   -0.123456
    for line in section.splitlines():
        m = re.match(r"\s*(\d+)\s+([A-Za-z]+)\s*:\s*([-+]?\d+\.\d+)", line)
        if m:
            idx = int(m.group(1))
            elem = m.group(2)
            chg = float(m.group(3))
            charges[idx] = (elem, chg)

    return charges

def main():
    text = Path(OUT_FILE).read_text(errors="ignore")
    elems = read_xyz_elements(XYZ_FILE)

    n_indices_zero = [i for i, e in enumerate(elems) if e.upper() == "N"]
    n_indices_one = [i + 1 for i in n_indices_zero]

    print("=" * 72)
    print("CMR_GOLD_058 corrected DFT extraction")
    print("=" * 72)

    final_energy = extract_final_energy(text)
    dipole = extract_dipole(text)

    print("Final single-point energy (Eh):", final_energy)
    print("Dipole moment (Debye):", dipole)
    print("N atom indices, zero-based:", n_indices_zero)
    print("N atom indices, one-based :", n_indices_one)
    print("")

    orbitals = extract_orbitals(text)

    if orbitals:
        occupied = [o for o in orbitals if o[1] > 0.0]
        virtual = [o for o in orbitals if o[1] == 0.0]

        homo = occupied[-1]
        lumo = virtual[0]

        gap = lumo[3] - homo[3]

        print("HOMO orbital no:", homo[0])
        print("HOMO energy (Eh):", homo[2])
        print("HOMO energy (eV):", homo[3])
        print("LUMO orbital no:", lumo[0])
        print("LUMO energy (Eh):", lumo[2])
        print("LUMO energy (eV):", lumo[3])
        print("Gap (eV):", gap)
    else:
        print("WARNING: Could not parse orbital table.")
        print("Use ORCA output around the ORBITAL ENERGIES section.")
    print("")

    mull = extract_atomic_charges(text, "MULLIKEN ATOMIC CHARGES")
    lowd = extract_atomic_charges(text, "LOEWDIN ATOMIC CHARGES")

    print("Mulliken N charges:")
    for idx in n_indices_zero:
        if idx in mull:
            print("  atom {} {}: {}".format(idx, mull[idx][0], mull[idx][1]))
        elif idx + 1 in mull:
            print("  atom {} {}: {}".format(idx + 1, mull[idx + 1][0], mull[idx + 1][1]))
        else:
            print("  N atom index {} not found in Mulliken table".format(idx))

    print("")

    print("Loewdin N charges:")
    for idx in n_indices_zero:
        if idx in lowd:
            print("  atom {} {}: {}".format(idx, lowd[idx][0], lowd[idx][1]))
        elif idx + 1 in lowd:
            print("  atom {} {}: {}".format(idx + 1, lowd[idx + 1][0], lowd[idx + 1][1]))
        else:
            print("  N atom index {} not found in Loewdin table".format(idx))

    print("=" * 72)

if __name__ == "__main__":
    main()