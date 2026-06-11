from pathlib import Path
import numpy as np

XYZ = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp\dft\molecules\CMR_GOLD_079\02_sp_charge\geom.xyz")

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "S": 1.05, "P": 1.07, "F": 0.57, "Cl": 1.02,
    "Br": 1.20, "I": 1.39,
}

def read_xyz(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = int(lines[0].strip())
    elements, coords = [], []
    for line in lines[2:2+n]:
        p = line.split()
        elements.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elements, np.array(coords, dtype=float)

def infer_bonds(elements, coords):
    bonds = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if elements[i] == "H" and elements[j] == "H":
                continue
            ri = COVALENT_RADII.get(elements[i], 0.76)
            rj = COVALENT_RADII.get(elements[j], 0.76)
            d = np.linalg.norm(coords[i] - coords[j])
            if 0.30 < d <= 1.20 * (ri + rj) + 0.08:
                bonds.append((i, j))
    return bonds

def build_graph(n, bonds):
    g = {i: set() for i in range(n)}
    for i, j in bonds:
        g[i].add(j)
        g[j].add(i)
    return g

def canonical_cycle(cyc):
    cyc = list(cyc)
    rotations = []
    for k in range(len(cyc)):
        r = cyc[k:] + cyc[:k]
        rotations.append(tuple(r))
        rotations.append(tuple(reversed(r)))
    return min(rotations)

def find_cycles_len_5_6(graph, elements):
    heavy = [i for i, e in enumerate(elements) if e != "H"]
    cycles = set()

    def dfs(start, current, path):
        if len(path) > 6:
            return
        for nb in graph[current]:
            if elements[nb] == "H":
                continue
            if nb == start and len(path) in (5, 6):
                cycles.add(canonical_cycle(path))
            elif nb not in path and nb >= start:
                dfs(start, nb, path + [nb])

    for start in heavy:
        dfs(start, start, [start])

    unique = []
    seen_sets = set()
    for cyc in sorted(cycles, key=lambda x: (len(x), x)):
        s = frozenset(cyc)
        if s not in seen_sets:
            seen_sets.add(s)
            unique.append(list(cyc))
    return unique

def plane_normal(points):
    center = points.mean(axis=0)
    x = points - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    distances = x @ normal
    rms = float(np.sqrt(np.mean(distances**2)))
    return normal, center, rms

def angle_between_normals(n1, n2):
    cosang = abs(float(np.dot(n1, n2)))
    cosang = min(1.0, max(-1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

elements, coords = read_xyz(XYZ)
bonds = infer_bonds(elements, coords)
graph = build_graph(len(elements), bonds)
cycles = find_cycles_len_5_6(graph, elements)

print("=" * 72)
print("CMR_GOLD_079 geometry diagnostic")
print("=" * 72)
print(f"XYZ file: {XYZ}")
print(f"Atoms: {len(elements)}")
print(f"Inferred bonds: {len(bonds)}")
print(f"Detected 5/6-membered heavy-atom rings: {len(cycles)}")
print()

ring_data = []
for idx, cyc in enumerate(cycles, 1):
    pts = coords[cyc]
    normal, center, rms = plane_normal(pts)
    labels = [f"{elements[i]}{i+1}" for i in cyc]
    ring_data.append((idx, cyc, normal, center, rms, labels))
    print(f"Ring {idx}:")
    print(f"  atoms      : {', '.join(labels)}")
    print(f"  size       : {len(cyc)}")
    print(f"  plane RMS  : {rms:.4f} Å")
    print(f"  center     : {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}")
    print()

print("=" * 72)
print("Pairwise ring-plane angles")
print("=" * 72)

for i in range(len(ring_data)):
    for j in range(i + 1, len(ring_data)):
        idx1, cyc1, n1, c1, rms1, labels1 = ring_data[i]
        idx2, cyc2, n2, c2, rms2, labels2 = ring_data[j]
        angle = angle_between_normals(n1, n2)
        dist = np.linalg.norm(c1 - c2)
        print(
            f"Ring {idx1} vs Ring {idx2}: "
            f"angle = {angle:6.2f} deg, "
            f"center distance = {dist:6.2f} Å"
        )

print("=" * 72)
print("Interpretation guide")
print("=" * 72)
print("< 10 deg  : nearly coplanar")
print("10-30 deg : mildly non-planar")
print("> 30 deg  : clearly non-planar / twisted")
print("> 45 deg  : strongly twisted geometry")