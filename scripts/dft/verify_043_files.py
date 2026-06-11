# -*- coding: utf-8 -*-
"""
Verify CMR_GOLD_043 structure/cube consistency.

Checks:
- XYZ atom count and formula
- cube header atom count and element list
- XYZ vs cube atom coordinate RMSD
- electron density cube vs ESP cube consistency
- inferred bond connectivity / molecular fragmentation
"""

import math
import numpy as np

XYZ_FILE = "CMR_GOLD_043_sp_charge_clean.xyz"
DENS_CUBE = "CMR_GOLD_043_sp_charge_clean.eldens.cube"
ESP_CUBE = "CMR_GOLD_043_sp_charge_clean.scfp.esp.cube"

BOHR_TO_ANG = 0.52917721092

ATOMIC_NUMBERS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "F": 0.57,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def safe_symbol(text):
    t = text.strip()
    if len(t) >= 2 and t[:2].capitalize() in ("Cl", "Br", "Si"):
        return t[:2].capitalize()
    return t[0].upper()


def read_xyz(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()

    n = int(lines[0].strip())
    elems = []
    coords = []

    for line in lines[2:2+n]:
        p = line.split()
        elems.append(safe_symbol(p[0]))
        coords.append([float(p[1]), float(p[2]), float(p[3])])

    return elems, np.array(coords, dtype=float)


def read_cube_atoms(path):
    with open(path, "r") as f:
        f.readline()
        f.readline()

        p = f.readline().split()
        natoms = abs(int(float(p[0])))

        # grid lines
        f.readline()
        f.readline()
        f.readline()

        elems = []
        coords = []

        for _ in range(natoms):
            p = f.readline().split()
            anum = int(float(p[0]))
            sym = ATOMIC_NUMBERS.get(anum, "X")
            x = float(p[2])
            y = float(p[3])
            z = float(p[4])
            elems.append(sym)
            coords.append([x, y, z])

    return elems, np.array(coords, dtype=float)


def formula(elems):
    order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    counts = {}
    for e in elems:
        counts[e] = counts.get(e, 0) + 1

    parts = []
    for e in order:
        if e in counts:
            parts.append("{}{}".format(e, counts[e] if counts[e] > 1 else ""))

    for e in sorted(counts):
        if e not in order:
            parts.append("{}{}".format(e, counts[e] if counts[e] > 1 else ""))

    return "".join(parts)


def pairwise_distance_ratio(coords_a, coords_b):
    n = min(len(coords_a), len(coords_b))
    ratios = []

    for i in range(n):
        for j in range(i+1, n):
            da = np.linalg.norm(coords_a[i] - coords_a[j])
            db = np.linalg.norm(coords_b[i] - coords_b[j])
            if db > 1e-8:
                ratios.append(da / db)

    if not ratios:
        return None

    return float(np.median(ratios))


def best_coord_compare(xyz_coords, cube_coords):
    """
    ORCA cube atom coordinates may be in Bohr or Angstrom depending on source.
    Compare both possibilities.
    """
    if len(xyz_coords) != len(cube_coords):
        return None

    cube_as_ang_1 = cube_coords.copy()
    cube_as_ang_2 = cube_coords * BOHR_TO_ANG

    rmsd_1 = np.sqrt(np.mean(np.sum((xyz_coords - cube_as_ang_1) ** 2, axis=1)))
    rmsd_2 = np.sqrt(np.mean(np.sum((xyz_coords - cube_as_ang_2) ** 2, axis=1)))

    max_1 = np.max(np.linalg.norm(xyz_coords - cube_as_ang_1, axis=1))
    max_2 = np.max(np.linalg.norm(xyz_coords - cube_as_ang_2, axis=1))

    if rmsd_2 < rmsd_1:
        return "cube Bohr -> Angstrom", rmsd_2, max_2
    else:
        return "cube already Angstrom", rmsd_1, max_1


def infer_bonds(elems, coords):
    bonds = []
    for i in range(len(coords)):
        ri = COVALENT_RADII.get(elems[i], 0.76)
        for j in range(i+1, len(coords)):
            rj = COVALENT_RADII.get(elems[j], 0.76)
            d = np.linalg.norm(coords[i] - coords[j])

            if elems[i] == "H" and elems[j] == "H":
                continue

            threshold = 1.20 * (ri + rj) + 0.08

            if 0.30 < d <= threshold:
                bonds.append((i, j, d, elems[i] + "-" + elems[j]))

    return bonds


def connected_components(n, bonds):
    adj = [[] for _ in range(n)]
    for i, j, _, _ in bonds:
        adj[i].append(j)
        adj[j].append(i)

    seen = set()
    comps = []

    for i in range(n):
        if i in seen:
            continue

        stack = [i]
        seen.add(i)
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        comps.append(comp)

    comps.sort(key=len, reverse=True)
    return comps


def main():
    xyz_elems, xyz_coords = read_xyz(XYZ_FILE)
    dens_elems, dens_coords = read_cube_atoms(DENS_CUBE)
    esp_elems, esp_coords = read_cube_atoms(ESP_CUBE)

    print("=" * 72)
    print("CMR_GOLD_043 FILE CONSISTENCY CHECK")
    print("=" * 72)

    print("XYZ file      :", XYZ_FILE)
    print("Density cube  :", DENS_CUBE)
    print("ESP cube      :", ESP_CUBE)
    print("")

    print("Atom counts:")
    print("  XYZ      :", len(xyz_elems))
    print("  dens cube:", len(dens_elems))
    print("  ESP cube :", len(esp_elems))
    print("")

    print("Formula:")
    print("  XYZ      :", formula(xyz_elems))
    print("  dens cube:", formula(dens_elems))
    print("  ESP cube :", formula(esp_elems))
    print("")

    print("Element sequence matches:")
    print("  XYZ vs dens cube:", xyz_elems == dens_elems)
    print("  XYZ vs ESP cube :", xyz_elems == esp_elems)
    print("  dens vs ESP cube:", dens_elems == esp_elems)
    print("")

    dens_cmp = best_coord_compare(xyz_coords, dens_coords)
    esp_cmp = best_coord_compare(xyz_coords, esp_coords)

    print("Coordinate agreement:")
    if dens_cmp:
        print("  XYZ vs dens cube: mode={}, RMSD={:.6f} A, max dev={:.6f} A".format(
            dens_cmp[0], dens_cmp[1], dens_cmp[2]
        ))
    if esp_cmp:
        print("  XYZ vs ESP cube : mode={}, RMSD={:.6f} A, max dev={:.6f} A".format(
            esp_cmp[0], esp_cmp[1], esp_cmp[2]
        ))
    print("")

    bonds = infer_bonds(xyz_elems, xyz_coords)
    comps = connected_components(len(xyz_elems), bonds)

    print("Connectivity from XYZ:")
    print("  inferred bonds:", len(bonds))
    print("  connected components:", len(comps))
    print("  component sizes:", [len(c) for c in comps])

    if len(comps) == 1:
        print("  Connectivity check: PASS - molecule is one connected component")
    else:
        print("  Connectivity check: WARNING - molecule appears fragmented")
        for k, comp in enumerate(comps, 1):
            labels = ["{}{}".format(xyz_elems[i], i+1) for i in comp]
            print("    component {}: {}".format(k, ", ".join(labels)))

    print("")
    print("Shortest/longest inferred bonds:")
    if bonds:
        bonds_sorted = sorted(bonds, key=lambda x: x[2])
        print("  shortest:")
        for i, j, d, typ in bonds_sorted[:5]:
            print("    {}{}-{}{}  {:.3f} A  {}".format(xyz_elems[i], i+1, xyz_elems[j], j+1, d, typ))
        print("  longest:")
        for i, j, d, typ in bonds_sorted[-5:]:
            print("    {}{}-{}{}  {:.3f} A  {}".format(xyz_elems[i], i+1, xyz_elems[j], j+1, d, typ))

    print("=" * 72)


if __name__ == "__main__":
    main()