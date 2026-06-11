# -*- coding: utf-8 -*-
"""
Generate Figure S5: Auxiliary molecular electrostatic potential maps
for CMR_GOLD_055 and CMR_GOLD_079.

Repository-ready version.

Key feature:
    CMR_GOLD_079 is automatically reoriented so that its principal
    molecular axis is directed toward the camera. This prevents the
    elongated multi-N molecule from appearing as an overly horizontal
    strip in the final SI figure.

Inputs:
    dft/molecules/CMR_GOLD_055/02_sp_charge/geom.xyz
    dft/molecules/CMR_GOLD_055/03_mep_cube/CMR_GOLD_055_sp_charge_clean.eldens.cube
    dft/molecules/CMR_GOLD_055/03_mep_cube/CMR_GOLD_055_sp_charge_clean.scfp.esp.cube

    dft/molecules/CMR_GOLD_079/02_sp_charge/geom.xyz
    dft/molecules/CMR_GOLD_079/03_mep_cube/CMR_GOLD_079_sp_charge_clean.eldens.cube
    dft/molecules/CMR_GOLD_079/03_mep_cube/CMR_GOLD_079_sp_charge_clean.scfp.esp.cube

Outputs:
    figures/supporting/Figure_S5_auxiliary_MEP_maps.png
    figures/supporting/Figure_S5_auxiliary_MEP_maps.tiff
    figures/supporting/Figure_S5_auxiliary_MEP_maps.pdf
    data/processed/Figure_S5_auxiliary_MEP_maps_summary.txt

Recommended environment:
    conda activate mepfig

Required packages:
    numpy
    scipy
    matplotlib
    pyvista
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# Project paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "figures" / "supporting"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Figure settings
# ============================================================

SURFACE_ISOVALUE = 0.004
GLOBAL_CLIP = 0.080
ESP_SMOOTH_SIGMA = 0.80

OUTPUT_PNG = FIG_DIR / "Figure_S5_auxiliary_MEP_maps.png"
OUTPUT_TIFF = FIG_DIR / "Figure_S5_auxiliary_MEP_maps.tiff"
OUTPUT_PDF = FIG_DIR / "Figure_S5_auxiliary_MEP_maps.pdf"
OUTPUT_SUMMARY = DATA_DIR / "Figure_S5_auxiliary_MEP_maps_summary.txt"

FINAL_DPI = 600
SCREENSHOT_SIZE = (2200, 1500)

FIG_WIDTH = 15.8
FIG_HEIGHT = 6.0

PANEL_LETTER_X = 0.020
PANEL_LETTER_Y = 0.955
PANEL_TITLE_Y = 0.985

SURFACE_OPACITY = 0.62
SURFACE_AMBIENT = 0.26
SURFACE_DIFFUSE = 0.88
SURFACE_SPECULAR = 0.12
SURFACE_SPECULAR_POWER = 18

ATOM_OPACITY = 0.94
BOND_OPACITY = 0.86
ATOM_RADIUS_HEAVY = 0.34
ATOM_RADIUS_H = 0.22
BOND_RADIUS_HEAVY = 0.105
BOND_RADIUS_H = 0.078

CAMERA_MARGIN = 1.08
CAMERA_Z_FACTOR = 2.55
CAMERA_X_TILT = 0.08
CAMERA_Y_TILT = -0.05

BOHR_TO_ANG = 0.52917721092

# The target direction is the viewing direction used by the camera.
# Aligning the long molecular axis to this vector turns the molecule
# toward the viewer instead of leaving it horizontally extended.
VIEW_DIRECTION = np.array([0.55, 0.10, 0.83], dtype=float)
VIEW_DIRECTION = VIEW_DIRECTION / np.linalg.norm(VIEW_DIRECTION)

# Molecule-specific orientation control.
# mode:
#   "none"              : no special orientation
#   "long_axis_to_view" : align principal molecular axis to VIEW_DIRECTION
#
# post_rot_* values are applied after PCA alignment for mild visual tuning.
ORIENTATION = {
    "CMR_GOLD_055": {
        "mode": "none",
        "post_rot_x": 0.0,
        "post_rot_y": 0.0,
        "post_rot_z": 0.0,
    },
    "CMR_GOLD_079": {
        "mode": "long_axis_to_view",
        "post_rot_x": 4.0,
        "post_rot_y": 0.0,
        "post_rot_z": -8.0,
    },
}


# ============================================================
# Input molecules
# ============================================================

MOLECULES = [
    {
        "compound_id": "CMR_GOLD_055",
        "panel_label": "A",
        "title": "CMR-GOLD-055",
        "structure": PROJECT_ROOT / "dft" / "molecules" / "CMR_GOLD_055" / "02_sp_charge" / "geom.xyz",
        "density_cube": PROJECT_ROOT / "dft" / "molecules" / "CMR_GOLD_055" / "03_mep_cube" / "CMR_GOLD_055_sp_charge_clean.eldens.cube",
        "esp_cube": PROJECT_ROOT / "dft" / "molecules" / "CMR_GOLD_055" / "03_mep_cube" / "CMR_GOLD_055_sp_charge_clean.scfp.esp.cube",
        "expected_components": 1,
    },
    {
        "compound_id": "CMR_GOLD_079",
        "panel_label": "B",
        "title": "CMR-GOLD-079",
        "structure": PROJECT_ROOT / "dft" / "molecules" / "CMR_GOLD_079" / "02_sp_charge" / "geom.xyz",
        "density_cube": PROJECT_ROOT / "dft" / "molecules" / "CMR_GOLD_079" / "03_mep_cube" / "CMR_GOLD_079_sp_charge_clean.eldens.cube",
        "esp_cube": PROJECT_ROOT / "dft" / "molecules" / "CMR_GOLD_079" / "03_mep_cube" / "CMR_GOLD_079_sp_charge_clean.scfp.esp.cube",
        "expected_components": 1,
    },
]


# Negative ESP = red; positive ESP = blue.
MEP_CMAP = LinearSegmentedColormap.from_list(
    "coumarin-logp_RWB",
    [
        "#D90429",
        "#F06C7B",
        "#F7F7F7",
        "#75AADB",
        "#0066CC",
    ],
    N=512,
)

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

ATOM_COLORS = {
    "H": "#F8F8F8",
    "C": "#A8B5AD",
    "N": "#3F83E8",
    "O": "#EF4444",
    "S": "#DCCB5F",
    "P": "#F59E0B",
    "F": "#6FD96F",
    "Cl": "#59C36A",
    "Br": "#A97142",
    "I": "#7B4FA2",
}

pv.OFF_SCREEN = True


# ============================================================
# File checks
# ============================================================

def check_files() -> None:
    missing = []

    for mol in MOLECULES:
        for key in ("structure", "density_cube", "esp_cube"):
            path = Path(mol[key])
            if not path.exists():
                missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  - " + "\n  - ".join(missing)
        )


# ============================================================
# Structure readers
# ============================================================

def safe_symbol(text: str) -> str:
    if not text:
        return "C"

    token = text.strip()

    if not token:
        return "C"

    if len(token) >= 2 and token[:2].capitalize() in ("Cl", "Br"):
        return token[:2].capitalize()

    return token[0].upper()


def read_xyz(xyz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = xyz_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    natoms = int(lines[0].strip())
    elements = []
    coords = []

    for line in lines[2:2 + natoms]:
        parts = line.split()

        if len(parts) < 4:
            continue

        elements.append(safe_symbol(parts[0]))
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(coords) != natoms:
        raise ValueError(
            f"XYZ atom count mismatch in {xyz_path}. "
            f"Expected {natoms}, got {len(coords)}."
        )

    return np.array(elements, dtype=object), np.array(coords, dtype=float)


def read_pdb(pdb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    elements = []
    coords = []

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except Exception:
                continue

            element = line[76:78].strip()

            if not element:
                name = line[12:16].strip()
                letters = "".join(char for char in name if char.isalpha())
                element = safe_symbol(letters)
            else:
                element = safe_symbol(element)

            elements.append(element)
            coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No atoms could be read from PDB: {pdb_path}")

    return np.array(elements, dtype=object), np.array(coords, dtype=float)


def read_structure(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ext = path.suffix.lower()

    if ext == ".xyz":
        return read_xyz(path)

    if ext == ".pdb":
        return read_pdb(path)

    raise ValueError(f"Unsupported structure file format: {path}")


def formula_from_elements(elements: np.ndarray) -> str:
    order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    counts: Dict[str, int] = {}

    for element in elements:
        counts[element] = counts.get(element, 0) + 1

    parts = []

    for element in order:
        if element in counts:
            number = counts[element]
            parts.append(f"{element}{number if number > 1 else ''}")

    for element in sorted(counts):
        if element not in order:
            number = counts[element]
            parts.append(f"{element}{number if number > 1 else ''}")

    return "".join(parts)


# ============================================================
# Cube reader
# ============================================================

def read_cube(cube_path: Path) -> Dict[str, np.ndarray]:
    with cube_path.open("r", encoding="utf-8", errors="ignore") as handle:
        _comment1 = handle.readline()
        _comment2 = handle.readline()

        line = handle.readline().split()
        natoms = abs(int(float(line[0])))
        origin = np.array(
            [float(line[1]), float(line[2]), float(line[3])],
            dtype=float,
        )

        dims = []
        axes = []

        for _ in range(3):
            parts = handle.readline().split()
            n_points = abs(int(float(parts[0])))
            axis_vec = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=float,
            )
            dims.append(n_points)
            axes.append(axis_vec)

        atom_numbers = []
        atom_coords = []

        for _ in range(natoms):
            parts = handle.readline().split()
            atomic_number = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            atom_numbers.append(atomic_number)
            atom_coords.append([x, y, z])

        values = []

        for line in handle:
            if line.strip():
                values.extend(line.split())

    dims_tuple = tuple(dims)
    values_arr = np.array(values, dtype=np.float32)
    expected = dims_tuple[0] * dims_tuple[1] * dims_tuple[2]

    if values_arr.size != expected:
        raise ValueError(
            f"{cube_path}: cube data size mismatch. "
            f"Expected {expected}, got {values_arr.size}."
        )

    data = values_arr.reshape(dims_tuple, order="C")

    return {
        "origin": origin,
        "axes": np.array(axes, dtype=float),
        "dims": np.array(dims_tuple, dtype=int),
        "data": data,
        "atom_numbers": np.array(atom_numbers, dtype=int),
        "atom_coords": np.array(atom_coords, dtype=float),
    }


# ============================================================
# Unit harmonisation
# ============================================================

def guess_cube_to_angstrom_factor(
    cube_atom_coords: np.ndarray,
    structure_coords: np.ndarray,
) -> float:
    n_atoms = min(len(cube_atom_coords), len(structure_coords))

    if n_atoms < 2:
        return 1.0

    cube_distances = []
    structure_distances = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            cube_dist = np.linalg.norm(cube_atom_coords[i] - cube_atom_coords[j])
            struct_dist = np.linalg.norm(structure_coords[i] - structure_coords[j])

            if struct_dist > 1e-6:
                cube_distances.append(cube_dist)
                structure_distances.append(struct_dist)

    if not cube_distances:
        return 1.0

    ratio = np.median(np.array(cube_distances) / np.array(structure_distances))

    if 1.65 < ratio < 2.10:
        return BOHR_TO_ANG

    if 0.85 < ratio < 1.15:
        return 1.0

    return 1.0


def scale_cube_to_angstrom(
    cube_obj: Dict[str, np.ndarray],
    factor: float,
) -> Dict[str, np.ndarray]:
    cube_scaled = dict(cube_obj)
    cube_scaled["origin"] = cube_scaled["origin"] * factor
    cube_scaled["axes"] = cube_scaled["axes"] * factor
    cube_scaled["atom_coords"] = cube_scaled["atom_coords"] * factor
    return cube_scaled


# ============================================================
# Rotation / orientation utilities
# ============================================================

def normalise_vector(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)

    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    return vec / norm


def rotation_matrix_axis_angle(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = normalise_vector(axis)
    angle = np.deg2rad(angle_degrees)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    c1 = 1.0 - c

    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_vec = normalise_vector(source)
    target_vec = normalise_vector(target)

    cross = np.cross(source_vec, target_vec)
    dot = float(np.dot(source_vec, target_vec))

    if dot > 0.999999:
        return np.eye(3, dtype=float)

    if dot < -0.999999:
        # 180-degree rotation around any axis perpendicular to source.
        trial_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(source_vec, trial_axis)) > 0.90:
            trial_axis = np.array([0.0, 1.0, 0.0])

        axis = normalise_vector(np.cross(source_vec, trial_axis))
        return rotation_matrix_axis_angle(axis, 180.0)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )

    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def euler_rotation_matrix(
    rot_x: float = 0.0,
    rot_y: float = 0.0,
    rot_z: float = 0.0,
) -> np.ndarray:
    rx = rotation_matrix_axis_angle(np.array([1.0, 0.0, 0.0]), rot_x)
    ry = rotation_matrix_axis_angle(np.array([0.0, 1.0, 0.0]), rot_y)
    rz = rotation_matrix_axis_angle(np.array([0.0, 0.0, 1.0]), rot_z)

    # Apply X, then Y, then Z.
    return rz @ ry @ rx


def principal_molecular_axis(
    elements: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    heavy_mask = elements != "H"
    selected = coords[heavy_mask] if np.any(heavy_mask) else coords

    if len(selected) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    centered = selected - np.mean(selected, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    axis = vh[0]

    # Keep direction deterministic.
    if axis[0] < 0:
        axis = -axis

    return normalise_vector(axis)


def rotate_points(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    return (points - center) @ rotation_matrix.T + center


def rotate_surface(
    surface: pv.PolyData,
    rotation_matrix: np.ndarray,
    center: np.ndarray,
) -> pv.PolyData:
    rotated = surface.copy(deep=True)
    rotated.points = rotate_points(rotated.points, rotation_matrix, center)
    return safe_compute_normals(rotated)


def apply_molecule_orientation(
    compound_id: str,
    elements: np.ndarray,
    coords: np.ndarray,
    surface: pv.PolyData,
) -> Tuple[np.ndarray, pv.PolyData, Dict[str, str]]:
    orientation = ORIENTATION.get(compound_id, {"mode": "none"})

    mode = orientation.get("mode", "none")
    center = np.mean(coords, axis=0)

    rotation_total = np.eye(3, dtype=float)
    principal_axis_before = principal_molecular_axis(elements, coords)

    if mode == "long_axis_to_view":
        rotation_total = rotation_matrix_from_vectors(
            principal_axis_before,
            VIEW_DIRECTION,
        ) @ rotation_total

    elif mode == "manual_euler":
        pass

    elif mode == "none":
        pass

    else:
        raise ValueError(f"Unsupported orientation mode for {compound_id}: {mode}")

    post_rotation = euler_rotation_matrix(
        rot_x=float(orientation.get("post_rot_x", 0.0)),
        rot_y=float(orientation.get("post_rot_y", 0.0)),
        rot_z=float(orientation.get("post_rot_z", 0.0)),
    )

    rotation_total = post_rotation @ rotation_total

    coords_rotated = rotate_points(coords, rotation_total, center)
    surface_rotated = rotate_surface(surface, rotation_total, center)

    principal_axis_after = principal_molecular_axis(elements, coords_rotated)

    info = {
        "orientation_mode": mode,
        "principal_axis_before": np.array2string(
            principal_axis_before,
            precision=5,
            separator=", ",
        ),
        "principal_axis_after": np.array2string(
            principal_axis_after,
            precision=5,
            separator=", ",
        ),
        "post_rotation_degrees": (
            f"x={orientation.get('post_rot_x', 0.0)}, "
            f"y={orientation.get('post_rot_y', 0.0)}, "
            f"z={orientation.get('post_rot_z', 0.0)}"
        ),
    }

    return coords_rotated, surface_rotated, info


# ============================================================
# Grid and surface processing
# ============================================================

def cube_to_imagedata(
    data3d: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    scalar_name: str,
) -> pv.ImageData:
    spacing = np.array(
        [
            np.linalg.norm(axes[0]),
            np.linalg.norm(axes[1]),
            np.linalg.norm(axes[2]),
        ],
        dtype=float,
    )

    grid = pv.ImageData()
    grid.dimensions = np.array(data3d.shape, dtype=int)
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data[scalar_name] = data3d.ravel(order="F")
    return grid


def safe_compute_normals(poly: pv.PolyData) -> pv.PolyData:
    try:
        return poly.compute_normals(
            auto_orient_normals=True,
            consistent_normals=True,
            inplace=False,
        )
    except TypeError:
        try:
            return poly.compute_normals(auto_orient_normals=True, inplace=False)
        except TypeError:
            return poly.compute_normals(inplace=False)


def smooth_surface(poly: pv.PolyData) -> pv.PolyData:
    try:
        return poly.smooth_taubin(n_iter=42, pass_band=0.075)
    except Exception:
        try:
            return poly.smooth(
                n_iter=65,
                relaxation_factor=0.010,
                feature_smoothing=False,
                boundary_smoothing=True,
            )
        except Exception:
            return poly


def build_density_surface(
    density_cube: Dict[str, np.ndarray],
    isovalue: float,
) -> pv.PolyData:
    grid = cube_to_imagedata(
        density_cube["data"],
        density_cube["origin"],
        density_cube["axes"],
        "density",
    )

    surface = grid.contour(isosurfaces=[isovalue], scalars="density")

    if surface.n_points == 0:
        raise ValueError(f"No surface generated at isovalue {isovalue}.")

    surface = surface.triangulate()
    surface = surface.clean()
    surface = smooth_surface(surface)
    surface = safe_compute_normals(surface)
    return surface


def build_adjacency(mesh: pv.PolyData) -> List[set]:
    neighbors = [set() for _ in range(mesh.n_points)]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    for triangle in faces:
        a, b, c = int(triangle[0]), int(triangle[1]), int(triangle[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    return neighbors


def smooth_scalars_on_surface(
    mesh: pv.PolyData,
    values: np.ndarray,
    iterations: int = 4,
    alpha: float = 0.45,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)

    if not np.any(finite):
        values[:] = 0.0
        return values

    fill = np.nanmedian(values[finite])
    values[~finite] = fill
    neighbors = build_adjacency(mesh)

    for _ in range(iterations):
        new_values = values.copy()

        for i, nb in enumerate(neighbors):
            if not nb:
                continue

            nb_idx = list(nb)
            nb_mean = values[nb_idx].mean()
            new_values[i] = (1.0 - alpha) * values[i] + alpha * nb_mean

        values = new_values

    return values


def sample_esp_onto_surface(
    surface: pv.PolyData,
    esp_cube: Dict[str, np.ndarray],
    smooth_iterations: int = 4,
) -> Tuple[pv.PolyData, np.ndarray]:
    esp_data = gaussian_filter(esp_cube["data"], sigma=ESP_SMOOTH_SIGMA)

    esp_grid = cube_to_imagedata(
        esp_data,
        esp_cube["origin"],
        esp_cube["axes"],
        "esp",
    )

    sampled = surface.sample(esp_grid)

    if "esp" not in sampled.point_data:
        raise RuntimeError("ESP sampling onto surface failed.")

    esp = np.asarray(sampled.point_data["esp"], dtype=float)

    if np.any(~np.isfinite(esp)):
        finite = np.isfinite(esp)
        fill = np.nanmedian(esp[finite]) if np.any(finite) else 0.0
        esp = np.where(np.isfinite(esp), esp, fill)

    if smooth_iterations > 0:
        esp = smooth_scalars_on_surface(
            sampled,
            esp,
            iterations=smooth_iterations,
            alpha=0.45,
        )

    esp = np.clip(esp, -GLOBAL_CLIP, GLOBAL_CLIP)
    sampled.point_data["esp"] = esp

    return sampled, esp


# ============================================================
# Bonds and molecule rendering
# ============================================================

def infer_bonds(elements: np.ndarray, coords: np.ndarray) -> List[Tuple[int, int]]:
    bonds = []
    n_atoms = len(coords)

    for i in range(n_atoms):
        ri = COVALENT_RADII.get(elements[i], 0.76)

        for j in range(i + 1, n_atoms):
            rj = COVALENT_RADII.get(elements[j], 0.76)
            distance = np.linalg.norm(coords[i] - coords[j])
            threshold = 1.20 * (ri + rj) + 0.08

            if elements[i] == "H" and elements[j] == "H":
                continue

            if 0.30 < distance <= threshold:
                bonds.append((i, j))

    return bonds


def connected_components(
    n_atoms: int,
    bonds: List[Tuple[int, int]],
) -> List[List[int]]:
    adjacency = [[] for _ in range(n_atoms)]

    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    seen = set()
    components = []

    for i in range(n_atoms):
        if i in seen:
            continue

        stack = [i]
        seen.add(i)
        component = []

        while stack:
            node = stack.pop()
            component.append(node)

            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    components.sort(key=len, reverse=True)
    return components


def validate_connectivity(
    title: str,
    elements: np.ndarray,
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    expected_components: int = 1,
) -> List[List[int]]:
    components = connected_components(len(elements), bonds)

    if len(components) != expected_components:
        labels = []

        for idx, component in enumerate(components, 1):
            atom_labels = [f"{elements[i]}{i + 1}" for i in component]
            labels.append(f"component {idx}: {', '.join(atom_labels)}")

        raise RuntimeError(
            f"{title} connectivity failed. "
            f"Expected {expected_components} component(s), got {len(components)}.\n"
            + "\n".join(labels)
        )

    return components


def get_atom_color(symbol: str) -> str:
    return ATOM_COLORS.get(symbol, "#B8B8B8")


def get_atom_radius(symbol: str) -> float:
    return ATOM_RADIUS_H if symbol == "H" else ATOM_RADIUS_HEAVY


def get_bond_radius(symbol_1: str, symbol_2: str) -> float:
    if symbol_1 == "H" or symbol_2 == "H":
        return BOND_RADIUS_H

    return BOND_RADIUS_HEAVY


def add_internal_molecule(
    plotter: pv.Plotter,
    elements: np.ndarray,
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
) -> None:
    for i, j in bonds:
        p1 = coords[i]
        p2 = coords[j]

        line = pv.Line(p1, p2, resolution=1)
        tube = line.tube(radius=get_bond_radius(elements[i], elements[j]))

        plotter.add_mesh(
            tube,
            color="#D6D6D6",
            opacity=BOND_OPACITY,
            smooth_shading=True,
            ambient=0.22,
            diffuse=0.82,
            specular=0.16,
            specular_power=18,
            show_scalar_bar=False,
        )

    for symbol, xyz in zip(elements, coords):
        sphere = pv.Sphere(
            radius=get_atom_radius(symbol),
            center=xyz,
            theta_resolution=32,
            phi_resolution=32,
        )

        plotter.add_mesh(
            sphere,
            color=get_atom_color(symbol),
            opacity=ATOM_OPACITY,
            smooth_shading=True,
            ambient=0.22,
            diffuse=0.84,
            specular=0.22,
            specular_power=24,
            show_scalar_bar=False,
        )


# ============================================================
# Plotter and camera
# ============================================================

def setup_plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=SCREENSHOT_SIZE)
    plotter.set_background("white")

    try:
        plotter.enable_anti_aliasing()
    except Exception:
        pass

    try:
        plotter.disable_depth_peeling()
    except Exception:
        pass

    try:
        plotter.remove_all_lights()
    except Exception:
        pass

    center = (0.0, 0.0, 0.0)

    lights = [
        pv.Light(
            position=(5.5, 5.0, 12.0),
            focal_point=center,
            color="white",
            intensity=1.20,
        ),
        pv.Light(
            position=(-8.0, -4.0, 8.0),
            focal_point=center,
            color="white",
            intensity=0.55,
        ),
        pv.Light(
            position=(0.0, -8.0, 4.0),
            focal_point=center,
            color="white",
            intensity=0.30,
        ),
    ]

    for light in lights:
        plotter.add_light(light)

    return plotter


def combined_bounds(surface: pv.PolyData, coords: np.ndarray) -> Tuple[float, ...]:
    combined = np.vstack([surface.points, coords])

    return (
        np.min(combined[:, 0]),
        np.max(combined[:, 0]),
        np.min(combined[:, 1]),
        np.max(combined[:, 1]),
        np.min(combined[:, 2]),
        np.max(combined[:, 2]),
    )


def bounds_center(bounds: Tuple[float, ...]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    return np.array(
        [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ],
        dtype=float,
    )


def bounds_spans(bounds: Tuple[float, ...]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    return np.array(
        [
            xmax - xmin,
            ymax - ymin,
            zmax - zmin,
        ],
        dtype=float,
    )


def compute_common_camera_scale(prepared_molecules: List[Dict]) -> float:
    aspect = SCREENSHOT_SIZE[0] / SCREENSHOT_SIZE[1]
    required_scales = []

    for mol in prepared_molecules:
        spans = bounds_spans(mol["bounds"])
        x_span = spans[0]
        y_span = spans[1]

        scale_y = 0.5 * y_span
        scale_x = 0.5 * x_span / aspect

        required_scales.append(max(scale_x, scale_y))

    return CAMERA_MARGIN * max(required_scales)


def apply_camera(
    plotter: pv.Plotter,
    bounds: Tuple[float, ...],
    common_parallel_scale: float,
) -> None:
    center = bounds_center(bounds)
    spans = bounds_spans(bounds)
    span = max(np.max(spans), 1.0)

    cam_pos = center + np.array(
        [
            CAMERA_X_TILT * span,
            CAMERA_Y_TILT * span,
            CAMERA_Z_FACTOR * span,
        ],
        dtype=float,
    )

    plotter.camera.position = tuple(cam_pos)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = common_parallel_scale


def render_panel_image(mol_data: Dict, common_parallel_scale: float) -> np.ndarray:
    plotter = setup_plotter()

    surface = mol_data["surface"]
    elements = mol_data["elements"]
    coords = mol_data["coords"]
    bonds = mol_data["bonds"]

    plotter.add_mesh(
        surface,
        scalars="esp",
        cmap=MEP_CMAP,
        clim=[-GLOBAL_CLIP, GLOBAL_CLIP],
        opacity=SURFACE_OPACITY,
        show_edges=False,
        smooth_shading=True,
        ambient=SURFACE_AMBIENT,
        diffuse=SURFACE_DIFFUSE,
        specular=SURFACE_SPECULAR,
        specular_power=SURFACE_SPECULAR_POWER,
        lighting=True,
        show_scalar_bar=False,
    )

    try:
        plotter.remove_scalar_bar()
    except Exception:
        pass

    add_internal_molecule(plotter, elements, coords, bonds)
    apply_camera(plotter, mol_data["bounds"], common_parallel_scale)

    plotter.render()
    image = plotter.screenshot(return_img=True, transparent_background=False)
    plotter.close()

    return np.asarray(image)


# ============================================================
# Data preparation
# ============================================================

def prepare_molecule(cfg: Dict) -> Dict:
    compound_id = cfg["compound_id"]

    elements, coords = read_structure(Path(cfg["structure"]))
    formula = formula_from_elements(elements)

    if "expected_formula" in cfg and formula != cfg["expected_formula"]:
        raise RuntimeError(
            f"{cfg['title']} formula mismatch. "
            f"Expected {cfg['expected_formula']}, got {formula}."
        )

    density_cube = read_cube(Path(cfg["density_cube"]))
    esp_cube = read_cube(Path(cfg["esp_cube"]))

    factor = guess_cube_to_angstrom_factor(density_cube["atom_coords"], coords)
    density_cube = scale_cube_to_angstrom(density_cube, factor)
    esp_cube = scale_cube_to_angstrom(esp_cube, factor)

    density_surface = build_density_surface(density_cube, SURFACE_ISOVALUE)

    sampled_surface, esp_surface = sample_esp_onto_surface(
        density_surface,
        esp_cube,
        smooth_iterations=4,
    )

    coords_oriented, surface_oriented, orientation_info = apply_molecule_orientation(
        compound_id,
        elements,
        coords,
        sampled_surface,
    )

    bonds = infer_bonds(elements, coords_oriented)
    components = validate_connectivity(
        cfg["title"],
        elements,
        coords_oriented,
        bonds,
        expected_components=cfg.get("expected_components", 1),
    )

    bounds = combined_bounds(surface_oriented, coords_oriented)

    return {
        "compound_id": compound_id,
        "panel_label": cfg["panel_label"],
        "title": cfg["title"],
        "elements": elements,
        "coords": coords_oriented,
        "bonds": bonds,
        "components": components,
        "formula": formula,
        "surface": surface_oriented,
        "esp_surface_values": esp_surface,
        "bounds": bounds,
        "structure": str(cfg["structure"]),
        "density_cube": str(cfg["density_cube"]),
        "esp_cube": str(cfg["esp_cube"]),
        "orientation_info": orientation_info,
    }


# ============================================================
# Final figure assembly
# ============================================================

def assemble_final_figure(
    panel_images: List[np.ndarray],
    molecules: List[Dict],
    output_png: Path,
    output_tiff: Path,
    output_pdf: Path,
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FINAL_DPI)

    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.00, 1.00, 0.052],
        wspace=0.030,
        left=0.020,
        right=0.965,
        top=0.925,
        bottom=0.070,
    )

    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    colorbar_axis = fig.add_subplot(grid[0, 2])

    for ax, image, mol in zip(axes, panel_images, molecules):
        ax.imshow(image)
        ax.axis("off")

        ax.text(
            PANEL_LETTER_X,
            PANEL_LETTER_Y,
            mol["panel_label"],
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

        ax.text(
            0.50,
            PANEL_TITLE_Y,
            mol["title"],
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            va="top",
            ha="center",
            color="black",
        )

    norm = Normalize(vmin=-GLOBAL_CLIP, vmax=GLOBAL_CLIP)
    scalar_map = ScalarMappable(norm=norm, cmap=MEP_CMAP)
    scalar_map.set_array([])

    colorbar = fig.colorbar(scalar_map, cax=colorbar_axis)
    colorbar.set_label("Electrostatic potential (a.u.)", fontsize=16, labelpad=14)
    colorbar.ax.tick_params(labelsize=12, width=1.2, length=5)
    colorbar.set_ticks(
        [-0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08]
    )

    for spine in colorbar.ax.spines.values():
        spine.set_linewidth(1.1)

    fig.patch.set_facecolor("white")

    fig.savefig(output_png, dpi=FINAL_DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(output_tiff, dpi=FINAL_DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(output_pdf, dpi=FINAL_DPI, facecolor="white", bbox_inches="tight")

    plt.close(fig)


def write_summary(
    molecules: List[Dict],
    common_parallel_scale: float,
) -> None:
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as handle:
        handle.write("Figure S5 auxiliary MEP rendering summary\n")
        handle.write("=" * 72 + "\n")
        handle.write(f"Surface isovalue: {SURFACE_ISOVALUE:.6f} a.u.\n")
        handle.write(f"Common ESP scale: -{GLOBAL_CLIP:.3f} to +{GLOBAL_CLIP:.3f} a.u.\n")
        handle.write(f"Common camera parallel scale: {common_parallel_scale:.6f}\n")
        handle.write(f"Output PNG: {OUTPUT_PNG}\n")
        handle.write(f"Output TIFF: {OUTPUT_TIFF}\n")
        handle.write(f"Output PDF: {OUTPUT_PDF}\n")
        handle.write("\n")

        for mol in molecules:
            values = mol["esp_surface_values"]
            orient = mol["orientation_info"]

            handle.write(f"{mol['title']}\n")
            handle.write(f"  formula                : {mol['formula']}\n")
            handle.write(f"  components             : {len(mol['components'])}\n")
            handle.write(f"  structure              : {mol['structure']}\n")
            handle.write(f"  density cube           : {mol['density_cube']}\n")
            handle.write(f"  esp cube               : {mol['esp_cube']}\n")
            handle.write(f"  orientation mode       : {orient['orientation_mode']}\n")
            handle.write(f"  principal axis before  : {orient['principal_axis_before']}\n")
            handle.write(f"  principal axis after   : {orient['principal_axis_after']}\n")
            handle.write(f"  post rotation degrees  : {orient['post_rotation_degrees']}\n")
            handle.write(f"  min ESP                : {np.min(values):.6f}\n")
            handle.write(f"  max ESP                : {np.max(values):.6f}\n")
            handle.write(f"  mean ESP               : {np.mean(values):.6f}\n")
            handle.write("\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 72)
    print("Preparing Figure S5 auxiliary MEP maps for CMR-GOLD-055 and CMR-GOLD-079")
    print("=" * 72)

    check_files()

    prepared = []

    for cfg in MOLECULES:
        print(f"Reading and preparing: {cfg['title']}")
        print(f"  structure: {cfg['structure']}")
        print(f"  density  : {cfg['density_cube']}")
        print(f"  esp      : {cfg['esp_cube']}")

        mol = prepare_molecule(cfg)
        prepared.append(mol)

        values = mol["esp_surface_values"]
        orient = mol["orientation_info"]

        print(f"  Formula: {mol['formula']}")
        print(f"  Connected components: {len(mol['components'])}")
        print(f"  Orientation mode: {orient['orientation_mode']}")
        print(f"  Principal axis before: {orient['principal_axis_before']}")
        print(f"  Principal axis after : {orient['principal_axis_after']}")
        print(
            "  Surface ESP stats: "
            f"min={np.min(values):.5f}, "
            f"max={np.max(values):.5f}, "
            f"mean={np.mean(values):.5f}"
        )

    common_parallel_scale = compute_common_camera_scale(prepared)

    print("-" * 72)
    print(f"Common fixed ESP scale: +/- {GLOBAL_CLIP:.5f} a.u.")
    print(f"Common camera parallel scale: {common_parallel_scale:.5f}")
    print("Both panels rendered with the same physical scale.")
    print("-" * 72)

    panel_images = []

    for mol in prepared:
        print(f"Rendering panel: {mol['title']}")
        panel_images.append(render_panel_image(mol, common_parallel_scale))

    print("Assembling final PNG, TIFF, and PDF...")
    assemble_final_figure(
        panel_images,
        prepared,
        OUTPUT_PNG,
        OUTPUT_TIFF,
        OUTPUT_PDF,
    )

    write_summary(prepared, common_parallel_scale)

    print("=" * 72)
    print("Done.")
    print("Output written to:")
    print(f"  {OUTPUT_PNG}")
    print(f"  {OUTPUT_TIFF}")
    print(f"  {OUTPUT_PDF}")
    print(f"  {OUTPUT_SUMMARY}")
    print("=" * 72)


if __name__ == "__main__":
    main()