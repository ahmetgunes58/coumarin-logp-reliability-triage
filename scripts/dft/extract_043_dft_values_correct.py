# -*- coding: utf-8 -*-
"""
Extract HOMO, LUMO, gap, dipole, and N-atom charges
from corrected CMR_GOLD_043 ORCA output.
"""

import re

OUT_FILE = "CMR_GOLD_043_sp_charge_clean.out"
XYZ_FILE = "geom.xyz"


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def read_xyz_elements(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()

    n = int(lines[0].strip())
    elems = []

    for line in lines[2:2 + n]:
        parts = line.split()
        elems.append(parts[0])

    return elems


def extract_final_energy(text):
    vals = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", text)
    return float(vals[-1]) if vals else None


def extract_dipole(text):
    vals = re.findall(r"Magnitude\s+\(Debye\)\s*:\s*([-+]?\d+\.\d+)", text)
    return float(vals[-1]) if vals else None


def extract_orbitals(text):
    marker = "ORBITAL ENERGIES"
    positions = [m.start() for m in re.finditer(marker, text)]

    if not positions:
        return []

    start = positions[-1]

    end_markers = [
        "MULLIKEN",
        "LOEWDIN",
        "ORCA PROPERTY",
        "Total SCF time",
        "FINAL SINGLE POINT ENERGY",
    ]

    end = len(text)

    for em in end_markers:
        pos = text.find(em, start + len(marker))
        if pos != -1 and pos < end:
            end = pos

    section = text[start:end]
    orbitals = []

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

    end_markers = [
        "Sum of atomic charges",
        "LOEWDIN ATOMIC CHARGES",
        "MULLIKEN REDUCED",
        "LOEWDIN REDUCED",
        "Mayer",
        "ORCA PROPERTY",
        "FINAL SINGLE POINT ENERGY",
    ]

    end = len(text)

    for em in end_markers:
        pos = text.find(em, start + len(heading))
        if pos != -1 and pos < end:
            end = pos

    section = text[start:end]
    charges = {}

    for line in section.splitlines():
        m = re.match(r"\s*(\d+)\s+([A-Za-z]+)\s*:\s*([-+]?\d+\.\d+)", line)

        if m:
            idx = int(m.group(1))
            elem = m.group(2)
            chg = float(m.group(3))
            charges[idx] = (elem, chg)

    return charges


def main():
    text = read_file(OUT_FILE)
    elems = read_xyz_elements(XYZ_FILE)

    n_indices_zero = [i for i, e in enumerate(elems) if e.upper() == "N"]
    n_indices_one = [i + 1 for i in n_indices_zero]

    print("=" * 72)
    print("CMR_GOLD_043 corrected-final DFT extraction")
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

    print("")

    mull = extract_atomic_charges(text, "MULLIKEN ATOMIC CHARGES")
    lowd = extract_atomic_charges(text, "LOEWDIN ATOMIC CHARGES")

    print("Mulliken N charges:")
    for idx in n_indices_zero:
        if idx in mull:
            print("  atom {} {}: {}".format(idx, mull[idx][0], mull[idx][1]))
        else:
            print("  N atom index {} not found in Mulliken table".format(idx))

    print("")

    print("Loewdin N charges:")
    for idx in n_indices_zero:
        if idx in lowd:
            print("  atom {} {}: {}".format(idx, lowd[idx][0], lowd[idx][1]))
        else:
            print("  N atom index {} not found in Loewdin table".format(idx))

    print("=" * 72)


if __name__ == "__main__":
    main()
