# -*- coding: utf-8 -*-

import numpy as np

XYZ_FILE = "CMR_GOLD_043_correct_3D.xyz"

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


def read_xyz(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()

    n = int(lines[0].strip())
    elems = []
    coords = []

    for line in lines[2:2+n]:
        p = line.split()
        elems.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])

    return elems, np.array(coords, dtype=float)


def formula(elems):
    order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    counts = {}
    for e in elems:
        counts[e] = counts.get(e, 0) + 1

    out = []
    for e in order:
        if e in counts:
            out.append("{}{}".format(e, counts[e] if counts[e] > 1 else ""))

    return "".join(out)


def infer_bonds(elems, coords):
    bonds = []

    for i in range(len(coords)):
        ri = COVALENT_RADII.get(elems[i], 0.76)

        for j in range(i + 1, len(coords)):
            rj = COVALENT_RADII.get(elems[j], 0.76)
            d = np.linalg.norm(coords[i] - coords[j])

            if elems[i] == "H" and elems[j] == "H":
                continue

            threshold = 1.20 * (ri + rj) + 0.08

            if 0.30 < d <= threshold:
                bonds.append((i, j, d))

    return bonds


def connected_components(n, bonds):
    adj = [[] for _ in range(n)]
    for i, j, _ in bonds:
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
    elems, coords = read_xyz(XYZ_FILE)
    bonds = infer_bonds(elems, coords)
    comps = connected_components(len(elems), bonds)

    print("=" * 72)
    print("CMR_GOLD_043 corrected XYZ check")
    print("=" * 72)
    print("File:", XYZ_FILE)
    print("Atom count:", len(elems))
    print("Formula:", formula(elems))
    print("N count:", sum(1 for e in elems if e == "N"))
    print("O count:", sum(1 for e in elems if e == "O"))
    print("Inferred bonds:", len(bonds))
    print("Connected components:", len(comps))
    print("Component sizes:", [len(c) for c in comps])

    if len(comps) == 1:
        print("Connectivity check: PASS - molecule is one connected component")
    else:
        print("Connectivity check: WARNING - fragmented geometry")
        for k, comp in enumerate(comps, 1):
            labels = ["{}{}".format(elems[i], i + 1) for i in comp]
            print("Component {}: {}".format(k, ", ".join(labels)))

    print("=" * 72)


if __name__ == "__main__":
    main()