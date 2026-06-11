# -*- coding: utf-8 -*-
"""
Extract HOMO, LUMO, gap, dipole, and atom counts
from corrected CMR_GOLD_055 ORCA output.
"""

import re

OUT_FILE = "CMR_GOLD_055_sp_charge_clean.out"
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


def main():
    text = read_file(OUT_FILE)
    elems = read_xyz_elements(XYZ_FILE)

    print("=" * 72)
    print("CMR_GOLD_055 corrected DFT extraction")
    print("=" * 72)

    print("Atom count:", len(elems))
    print("N atom count:", sum(1 for e in elems if e.upper() == "N"))

    final_energy = extract_final_energy(text)
    dipole = extract_dipole(text)

    print("Final single-point energy (Eh):", final_energy)
    print("Dipole moment (Debye):", dipole)
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

    print("=" * 72)


if __name__ == "__main__":
    main()