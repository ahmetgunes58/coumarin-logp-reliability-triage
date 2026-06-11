# -*- coding: utf-8 -*-
"""
24e_make_figure7_four_panel_from_cubes.py

Final four-panel Figure 7 renderer for:
Electronic Non-Additivity Underlies Fragment-Based logP Misclassification
in Polar Nitrogen-Containing Coumarins.

Purpose
-------
Render a Q1-style, four-panel DFT/MEP diagnostic comparison:

A: CMR_GOLD_043 — compact N = 2 serviceable control
B: CMR_GOLD_044 — compact N = 2 serviceable control
C: CMR_GOLD_029 — FM1 polar overestimation case
D: CMR_GOLD_058 — D–A-conjugated severe case

The script uses local standardized XYZ + density cube + ESP cube files:

dft/molecules/<COMPOUND_ID>/geometry/<COMPOUND_ID>_sp.xyz
dft/molecules/<COMPOUND_ID>/cubes/<COMPOUND_ID>_density.cube
dft/molecules/<COMPOUND_ID>/cubes/<COMPOUND_ID>_esp.cube

Descriptor annotations are read from:
data/processed/Figure7_main_panel_descriptors.csv

If the descriptor CSV is missing, the script creates a filled default version and continues.

Outputs
-------
figures/manuscript/Figure_7_DFT_MEP_main_panel.png
figures/manuscript/Figure_7_DFT_MEP_main_panel.pdf
figures/manuscript/Figure_7_DFT_MEP_main_panel.tiff
figures/manuscript/Figure_7_DFT_MEP_main_panel_summary.txt

Run
---
conda activate mepfig
cd /d D:\\Makaleler\\coumarin-logp-working-source

python scripts\\24e_make_figure7_four_panel_from_cubes.py ^
  --project-root "D:\\Makaleler\\coumarin-logp-working-source" ^
  --descriptor-csv "D:\\Makaleler\\coumarin-logp-working-source\\data\\processed\\Figure7_main_panel_descriptors.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

pv.OFF_SCREEN = True

BOHR_TO_ANG = 0.52917721092

# Final manuscript/SI protocol
SURFACE_ISOVALUE = 0.004
GLOBAL_CLIP = 0.080
ESP_SMOOTH_SIGMA = 0.80

# Rendering
SCREENSHOT_SIZE = (2200, 1550)
FINAL_DPI = 600

# Final figure canvas
FIG_WIDTH = 14.8
FIG_HEIGHT = 11.4

# Surface rendering style
SURFACE_OPACITY = 0.62
SURFACE_AMBIENT = 0.26
SURFACE_DIFFUSE = 0.88
SURFACE_SPECULAR = 0.12
SURFACE_SPECULAR_POWER = 18

# Ball-and-stick rendering style
ATOM_OPACITY = 0.94
BOND_OPACITY = 0.86
ATOM_RADIUS_HEAVY = 0.34
ATOM_RADIUS_H = 0.22
BOND_RADIUS_HEAVY = 0.105
BOND_RADIUS_H = 0.078

# Camera style
CAMERA_MARGIN = 1.03
CAMERA_Z_FACTOR = 2.45
CAMERA_X_TILT = 0.14
CAMERA_Y_TILT = -0.08

# Negative ESP = red; positive ESP = blue.
MEP_CMAP = LinearSegmentedColormap.from_list(
    "coumarin-logp_RWB",
    ["#D90429", "#F06C7B", "#F7F7F7", "#75AADB", "#0066CC"],
    N=512,
)

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


DEFAULT_DESCRIPTOR_ROWS = [
    {
        "compound_id": "CMR_GOLD_043",
        "panel_label": "A",
        "role_label": "Compact N=2 serviceable control",
        "n_count": 2,
        "delta_logp_consensus": 0.17,
        "dipole_D": 5.7832,
        "gap_eV": 4.1157,
        "homo_eV": "",
        "lumo_eV": "",
    },
    {
        "compound_id": "CMR_GOLD_044",
        "panel_label": "B",
        "role_label": "Compact N=2 serviceable control",
        "n_count": 2,
        "delta_logp_consensus": -0.03,
        "dipole_D": 6.3502,
        "gap_eV": 4.1451,
        "homo_eV": -6.6834,
        "lumo_eV": -2.5383,
    },
    {
        "compound_id": "CMR_GOLD_029",
        "panel_label": "C",
        "role_label": "FM1 polar overestimation case",
        "n_count": 2,
        "delta_logp_consensus": -2.04,
        "dipole_D": 10.3695,
        "gap_eV": 4.4523,
        "homo_eV": "",
        "lumo_eV": "",
    },
    {
        "compound_id": "CMR_GOLD_058",
        "panel_label": "D",
        "role_label": "D–A-conjugated severe case",
        "n_count": 2,
        "delta_logp_consensus": -5.1936,
        "dipole_D": 6.1104,
        "gap_eV": 2.6080,
        "homo_eV": -4.9939,
        "lumo_eV": -2.3859,
    },
]

DESIRED_ORDER = ["CMR_GOLD_043", "CMR_GOLD_044", "CMR_GOLD_029", "CMR_GOLD_058"]


# =============================================================================
# BASIC READERS
# =============================================================================

def safe_symbol(text: str) -> str:
    if not text:
        return "C"

    token = str(text).strip()

    if not token:
        return "C"

    if len(token) >= 2 and token[:2].capitalize() in ("Cl", "Br"):
        return token[:2].capitalize()

    return token[0].upper()


def read_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")

    natoms = int(lines[0].strip())
    elements: List[str] = []
    coords: List[List[float]] = []

    for line in lines[2:2 + natoms]:
        parts = line.split()

        if len(parts) < 4:
            continue

        elements.append(safe_symbol(parts[0]))
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(coords) != natoms:
        raise ValueError(
            f"XYZ atom-count mismatch in {path}. Expected {natoms}, got {len(coords)}."
        )

    return np.array(elements, dtype=object), np.array(coords, dtype=float)


def read_cube(cube_path: Path) -> Dict[str, np.ndarray]:
    with cube_path.open("r", encoding="utf-8", errors="ignore") as f:
        _comment1 = f.readline()
        _comment2 = f.readline()

        line = f.readline().split()
        natoms = abs(int(float(line[0])))
        origin = np.array([float(line[1]), float(line[2]), float(line[3])], dtype=float)

        dims = []
        axes = []

        for _ in range(3):
            parts = f.readline().split()
            dims.append(abs(int(float(parts[0]))))
            axes.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float))

        atom_numbers = []
        atom_coords = []

        for _ in range(natoms):
            parts = f.readline().split()
            atom_numbers.append(int(float(parts[0])))
            atom_coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

        values = []

        for line in f:
            if line.strip():
                values.extend(line.split())

    dims_tuple = tuple(dims)
    values_array = np.array(values, dtype=np.float32)
    expected = dims_tuple[0] * dims_tuple[1] * dims_tuple[2]

    if values_array.size != expected:
        raise ValueError(
            f"{cube_path}: cube data-size mismatch. Expected {expected}, got {values_array.size}."
        )

    data = values_array.reshape(dims_tuple, order="C")

    return {
        "origin": origin,
        "axes": np.array(axes, dtype=float),
        "dims": np.array(dims, dtype=int),
        "data": data,
        "atom_numbers": np.array(atom_numbers, dtype=int),
        "atom_coords": np.array(atom_coords, dtype=float),
    }


# =============================================================================
# UNIT AND GRID HANDLING
# =============================================================================

def guess_cube_to_angstrom_factor(cube_atom_coords: np.ndarray, structure_coords: np.ndarray) -> float:
    n = min(len(cube_atom_coords), len(structure_coords))

    if n < 2:
        return 1.0

    cube_distances = []
    structure_distances = []

    for i in range(n):
        for j in range(i + 1, n):
            dc = np.linalg.norm(cube_atom_coords[i] - cube_atom_coords[j])
            ds = np.linalg.norm(structure_coords[i] - structure_coords[j])

            if ds > 1e-6:
                cube_distances.append(dc)
                structure_distances.append(ds)

    if not cube_distances:
        return 1.0

    ratio = np.median(np.array(cube_distances) / np.array(structure_distances))

    if 1.65 < ratio < 2.10:
        return BOHR_TO_ANG

    if 0.85 < ratio < 1.15:
        return 1.0

    return 1.0


def scale_cube_to_angstrom(cube_obj: Dict[str, np.ndarray], factor: float) -> Dict[str, np.ndarray]:
    cube_obj = dict(cube_obj)
    cube_obj["origin"] = cube_obj["origin"] * factor
    cube_obj["axes"] = cube_obj["axes"] * factor
    cube_obj["atom_coords"] = cube_obj["atom_coords"] * factor
    return cube_obj


def cube_to_imagedata(data3d: np.ndarray, origin: np.ndarray, axes: np.ndarray, scalar_name: str) -> pv.ImageData:
    spacing = np.array(
        [np.linalg.norm(axes[0]), np.linalg.norm(axes[1]), np.linalg.norm(axes[2])],
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


def build_density_surface(density_cube: Dict[str, np.ndarray], isovalue: float) -> pv.PolyData:
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

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    return neighbors


def smooth_scalars_on_surface(mesh: pv.PolyData, values: np.ndarray, iterations: int = 4, alpha: float = 0.45) -> np.ndarray:
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


def sample_esp_onto_surface(surface: pv.PolyData, esp_cube: Dict[str, np.ndarray], compound_id: str) -> Tuple[pv.PolyData, np.ndarray]:
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

    # Slightly stronger smoothing for the elongated CMR_GOLD_058 surface, following legacy rendering.
    smooth_iter = 7 if compound_id == "CMR_GOLD_058" else 4

    if smooth_iter > 0:
        esp = smooth_scalars_on_surface(sampled, esp, iterations=smooth_iter, alpha=0.45)

    esp = np.clip(esp, -GLOBAL_CLIP, GLOBAL_CLIP)
    sampled.point_data["esp"] = esp

    return sampled, esp


# =============================================================================
# MOLECULE RENDERING
# =============================================================================

def infer_bonds(elements: np.ndarray, coords: np.ndarray) -> List[Tuple[int, int]]:
    bonds = []
    n = len(coords)

    for i in range(n):
        ri = COVALENT_RADII.get(elements[i], 0.76)

        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(elements[j], 0.76)
            d = np.linalg.norm(coords[i] - coords[j])
            threshold = 1.20 * (ri + rj) + 0.08

            if elements[i] == "H" and elements[j] == "H":
                continue

            if 0.30 < d <= threshold:
                bonds.append((i, j))

    return bonds


def get_atom_color(sym: str) -> str:
    return ATOM_COLORS.get(sym, "#B8B8B8")


def get_atom_radius(sym: str) -> float:
    if sym == "H":
        return ATOM_RADIUS_H
    return ATOM_RADIUS_HEAVY


def get_bond_radius(sym1: str, sym2: str) -> float:
    if sym1 == "H" or sym2 == "H":
        return BOND_RADIUS_H
    return BOND_RADIUS_HEAVY


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

    lights = [
        pv.Light(position=(5.5, 5.0, 12.0), focal_point=(0, 0, 0), color="white", intensity=1.20),
        pv.Light(position=(-8.0, -4.0, 8.0), focal_point=(0, 0, 0), color="white", intensity=0.55),
        pv.Light(position=(0.0, -8.0, 4.0), focal_point=(0, 0, 0), color="white", intensity=0.30),
    ]

    for light in lights:
        plotter.add_light(light)

    return plotter


def add_internal_molecule(plotter: pv.Plotter, elements: np.ndarray, coords: np.ndarray, bonds: List[Tuple[int, int]]) -> None:
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

    for sym, xyz in zip(elements, coords):
        sphere = pv.Sphere(
            radius=get_atom_radius(sym),
            center=xyz,
            theta_resolution=32,
            phi_resolution=32,
        )

        plotter.add_mesh(
            sphere,
            color=get_atom_color(sym),
            opacity=ATOM_OPACITY,
            smooth_shading=True,
            ambient=0.22,
            diffuse=0.84,
            specular=0.22,
            specular_power=24,
            show_scalar_bar=False,
        )


def combined_bounds(surface: pv.PolyData, coords: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    combined = np.vstack([surface.points, coords])

    return (
        np.min(combined[:, 0]),
        np.max(combined[:, 0]),
        np.min(combined[:, 1]),
        np.max(combined[:, 1]),
        np.min(combined[:, 2]),
        np.max(combined[:, 2]),
    )


def bounds_center(bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    return np.array(
        [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)],
        dtype=float,
    )


def bounds_spans(bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    return np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)


def compute_common_camera_scale(prepared: List[Dict]) -> float:
    aspect = SCREENSHOT_SIZE[0] / SCREENSHOT_SIZE[1]
    required_scales = []

    for mol in prepared:
        spans = bounds_spans(mol["bounds"])
        x_span = spans[0]
        y_span = spans[1]
        scale_y = 0.5 * y_span
        scale_x = 0.5 * x_span / aspect
        required_scales.append(max(scale_x, scale_y))

    return CAMERA_MARGIN * max(required_scales)


def apply_camera(plotter: pv.Plotter, bounds: Tuple[float, float, float, float, float, float], common_parallel_scale: float) -> None:
    center = bounds_center(bounds)
    spans = bounds_spans(bounds)
    span = max(np.max(spans), 1.0)

    cam_pos = center + np.array(
        [CAMERA_X_TILT * span, CAMERA_Y_TILT * span, CAMERA_Z_FACTOR * span],
        dtype=float,
    )

    plotter.camera.position = tuple(cam_pos)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = common_parallel_scale


def render_panel_image(mol: Dict, common_parallel_scale: float) -> np.ndarray:
    plotter = setup_plotter()

    plotter.add_mesh(
        mol["surface"],
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

    add_internal_molecule(plotter, mol["elements"], mol["coords"], mol["bonds"])
    apply_camera(plotter, mol["bounds"], common_parallel_scale)

    plotter.render()
    img = plotter.screenshot(return_img=True, transparent_background=False)
    plotter.close()

    return np.asarray(img)


# =============================================================================
# DESCRIPTOR AND MOLECULE PREPARATION
# =============================================================================

def ensure_descriptor_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    df = pd.DataFrame(DEFAULT_DESCRIPTOR_ROWS)
    df.to_csv(path, index=False)
    print(f"[INFO] Descriptor CSV created with default values: {path}")


def load_descriptor_table(path: Path) -> pd.DataFrame:
    ensure_descriptor_csv(path)
    df = pd.read_csv(path)

    required_cols = {
        "compound_id",
        "panel_label",
        "role_label",
        "n_count",
        "delta_logp_consensus",
        "dipole_D",
        "gap_eV",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Descriptor CSV missing required columns: {sorted(missing)}")

    order_map = {cid: i for i, cid in enumerate(DESIRED_ORDER)}
    df = df[df["compound_id"].isin(DESIRED_ORDER)].copy()

    if len(df) != 4:
        present = set(df["compound_id"].tolist())
        absent = [cid for cid in DESIRED_ORDER if cid not in present]
        raise ValueError(f"Descriptor CSV does not contain all Figure 7 compounds. Missing: {absent}")

    df["__order__"] = df["compound_id"].map(order_map)
    df = df.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)

    return df


def find_structure_file(mol_dir: Path, compound_id: str) -> Path:
    candidates = [
        mol_dir / "geometry" / f"{compound_id}_sp.xyz",
        mol_dir / "geometry" / f"{compound_id}_opt.xyz",
        mol_dir / "input" / f"{compound_id}_sp.xyz",
        mol_dir / "input" / f"{compound_id}_optfreq.xyz",
        mol_dir / "input" / f"{compound_id}_start.xyz",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No suitable XYZ structure found for {compound_id}. Checked:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
    )


def prepare_molecule(row: pd.Series, project_root: Path) -> Dict:
    compound_id = str(row["compound_id"])
    mol_dir = project_root / "dft" / "molecules" / compound_id

    structure = find_structure_file(mol_dir, compound_id)
    density_cube = mol_dir / "cubes" / f"{compound_id}_density.cube"
    esp_cube = mol_dir / "cubes" / f"{compound_id}_esp.cube"

    required = [density_cube, esp_cube]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    elements, coords = read_xyz(structure)
    dens_cube_obj = read_cube(density_cube)
    esp_cube_obj = read_cube(esp_cube)

    factor = guess_cube_to_angstrom_factor(dens_cube_obj["atom_coords"], coords)
    dens_cube_obj = scale_cube_to_angstrom(dens_cube_obj, factor)
    esp_cube_obj = scale_cube_to_angstrom(esp_cube_obj, factor)

    surface = build_density_surface(dens_cube_obj, SURFACE_ISOVALUE)
    surface, esp_vals = sample_esp_onto_surface(surface, esp_cube_obj, compound_id=compound_id)

    bonds = infer_bonds(elements, coords)
    bnds = combined_bounds(surface, coords)

    return {
        "compound_id": compound_id,
        "panel_label": str(row["panel_label"]),
        "title": compound_id.replace("_", "-"),
        "role_label": str(row.get("role_label", "")),
        "n_count": row.get("n_count", np.nan),
        "delta_logp_consensus": row.get("delta_logp_consensus", np.nan),
        "dipole_D": row.get("dipole_D", np.nan),
        "gap_eV": row.get("gap_eV", np.nan),
        "homo_eV": row.get("homo_eV", np.nan),
        "lumo_eV": row.get("lumo_eV", np.nan),
        "elements": elements,
        "coords": coords,
        "bonds": bonds,
        "surface": surface,
        "esp_surface_values": esp_vals,
        "bounds": bnds,
        "structure": structure,
        "density_cube": density_cube,
        "esp_cube": esp_cube,
        "unit_factor": factor,
    }


# =============================================================================
# FINAL FIGURE ASSEMBLY
# =============================================================================

def fmt_float(value, digits: int = 2, signed: bool = False) -> str | None:
    try:
        if pd.isna(value):
            return None
        x = float(value)
    except Exception:
        return None

    if signed:
        return f"{x:+.{digits}f}"

    return f"{x:.{digits}f}"


def fmt_int(value) -> str | None:
    try:
        if pd.isna(value):
            return None
        return str(int(round(float(value))))
    except Exception:
        return None


def build_q1_annotation_text(mol: Dict) -> str:
    role = mol.get("role_label", "")
    n_text = fmt_int(mol.get("n_count", np.nan))
    dlogp = fmt_float(mol.get("delta_logp_consensus", np.nan), digits=2, signed=True)
    dipole = fmt_float(mol.get("dipole_D", np.nan), digits=2, signed=False)
    gap = fmt_float(mol.get("gap_eV", np.nan), digits=2, signed=False)

    lines = []

    if role:
        lines.append(role)

    metric_line = []

    if n_text is not None:
        metric_line.append(f"N = {n_text}")

    if dlogp is not None:
        metric_line.append(r"$\Delta$logP$_{cons}$ = " + dlogp)

    if metric_line:
        lines.append(" | ".join(metric_line))

    metric_line_2 = []

    if dipole is not None:
        metric_line_2.append(r"$\mu$ = " + f"{dipole} D")

    if gap is not None:
        metric_line_2.append(f"Gap = {gap} eV")

    if metric_line_2:
        lines.append(" | ".join(metric_line_2))

    return "\n".join(lines)


def assemble_figure(
    panel_images: List[np.ndarray],
    molecules: List[Dict],
    out_png: Path,
    out_pdf: Path,
    out_tiff: Path,
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FINAL_DPI)

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.055],
        height_ratios=[1.0, 1.0],
        wspace=0.028,
        hspace=0.080,
        left=0.026,
        right=0.965,
        top=0.955,
        bottom=0.055,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    cax = fig.add_subplot(gs[:, 2])

    for ax, img, mol in zip(axes, panel_images, molecules):
        ax.imshow(img)
        ax.axis("off")

        # Panel label: top-left
        ax.text(
            0.018,
            0.965,
            mol["panel_label"],
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

        # Compound ID: top-centre
        ax.text(
            0.50,
            0.990,
            mol["title"],
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
            ha="center",
            color="black",
        )

        # Q1-style compact annotation: bottom-centre, not bottom-left
        annotation = build_q1_annotation_text(mol)

        if annotation.strip():
            ax.text(
                0.50,
                0.058,
                annotation,
                transform=ax.transAxes,
                fontsize=9.4,
                va="bottom",
                ha="center",
                color="black",
                linespacing=1.13,
                bbox=dict(
                    boxstyle="square,pad=0.28",
                    facecolor="white",
                    edgecolor="#8F8F8F",
                    linewidth=0.75,
                    alpha=0.90,
                ),
                zorder=20,
            )

    norm = Normalize(vmin=-GLOBAL_CLIP, vmax=GLOBAL_CLIP)
    sm = ScalarMappable(norm=norm, cmap=MEP_CMAP)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Electrostatic potential (a.u.)", fontsize=14, labelpad=12)
    cb.ax.tick_params(labelsize=10.5, width=1.05, length=5)
    cb.set_ticks([-0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08])

    for spine in cb.ax.spines.values():
        spine.set_linewidth(1.0)

    fig.patch.set_facecolor("white")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=FINAL_DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(out_pdf, dpi=FINAL_DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(
        out_tiff,
        dpi=FINAL_DPI,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    plt.close(fig)


def write_summary(
    out_summary: Path,
    project_root: Path,
    descriptor_csv: Path,
    molecules: List[Dict],
    common_parallel_scale: float,
    output_files: Dict[str, Path],
) -> None:
    lines = []
    lines.append("Figure 7 four-panel DFT/MEP rendering summary")
    lines.append("=" * 78)
    lines.append(f"Project root: {project_root}")
    lines.append(f"Descriptor CSV: {descriptor_csv}")
    lines.append("")
    lines.append(f"Surface isovalue: {SURFACE_ISOVALUE:.6f} a.u.")
    lines.append(f"Common ESP scale: -{GLOBAL_CLIP:.3f} to +{GLOBAL_CLIP:.3f} a.u.")
    lines.append(f"ESP smoothing sigma: {ESP_SMOOTH_SIGMA:.3f}")
    lines.append(f"Common camera parallel scale: {common_parallel_scale:.6f}")
    lines.append(f"Screenshot size: {SCREENSHOT_SIZE}")
    lines.append(f"Final DPI: {FINAL_DPI}")
    lines.append("")
    lines.append("Outputs:")
    for label, path in output_files.items():
        lines.append(f"  {label}: {path}")
    lines.append("")
    lines.append("Panel files and ESP surface statistics:")
    lines.append("-" * 78)

    for mol in molecules:
        vals = mol["esp_surface_values"]

        lines.append(f"{mol['panel_label']} | {mol['title']}")
        lines.append(f"  role label   : {mol.get('role_label', '')}")
        lines.append(f"  structure    : {mol['structure']}")
        lines.append(f"  density cube : {mol['density_cube']}")
        lines.append(f"  esp cube     : {mol['esp_cube']}")
        lines.append(f"  unit factor  : {mol['unit_factor']:.8f}")
        lines.append(f"  atoms        : {len(mol['elements'])}")
        lines.append(f"  bonds        : {len(mol['bonds'])}")
        lines.append(f"  min ESP      : {np.min(vals):.6f}")
        lines.append(f"  max ESP      : {np.max(vals):.6f}")
        lines.append(f"  mean ESP     : {np.mean(vals):.6f}")
        lines.append("")

    out_summary.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render final four-panel Figure 7 DFT/MEP main panel."
    )

    parser.add_argument(
        "--project-root",
        required=True,
        help="Project root folder, e.g. D:\\Makaleler\\coumarin-logp-working-source",
    )

    parser.add_argument(
        "--descriptor-csv",
        required=True,
        help="Path to Figure7_main_panel_descriptors.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    descriptor_csv = Path(args.descriptor_csv)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    df = load_descriptor_table(descriptor_csv)

    print("=" * 78)
    print("Preparing final four-panel Figure 7")
    print("=" * 78)
    print(f"Project root: {project_root}")
    print(f"Descriptor CSV: {descriptor_csv}")
    print(f"Surface isovalue: {SURFACE_ISOVALUE:.6f} a.u.")
    print(f"Common ESP scale: -{GLOBAL_CLIP:.3f} to +{GLOBAL_CLIP:.3f} a.u.")
    print("-" * 78)

    prepared = []

    for _, row in df.iterrows():
        compound_id = row["compound_id"]
        print(f"Preparing {compound_id}...")
        mol = prepare_molecule(row, project_root)
        vals = mol["esp_surface_values"]
        print(
            f"  ESP stats: min={np.min(vals):.5f}, max={np.max(vals):.5f}, mean={np.mean(vals):.5f}"
        )
        prepared.append(mol)

    common_parallel_scale = compute_common_camera_scale(prepared)
    print("-" * 78)
    print(f"Common camera parallel scale: {common_parallel_scale:.6f}")
    print("-" * 78)

    panel_images = []

    for mol in prepared:
        print(f"Rendering {mol['title']}...")
        img = render_panel_image(mol, common_parallel_scale)
        panel_images.append(img)

    out_dir = project_root / "figures" / "manuscript"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "Figure_7_DFT_MEP_main_panel.png"
    out_pdf = out_dir / "Figure_7_DFT_MEP_main_panel.pdf"
    out_tiff = out_dir / "Figure_7_DFT_MEP_main_panel.tiff"
    out_summary = out_dir / "Figure_7_DFT_MEP_main_panel_summary.txt"

    print("Assembling final figure...")
    assemble_figure(panel_images, prepared, out_png, out_pdf, out_tiff)

    output_files = {
        "PNG": out_png,
        "PDF": out_pdf,
        "TIFF": out_tiff,
        "SUMMARY": out_summary,
    }

    write_summary(
        out_summary=out_summary,
        project_root=project_root,
        descriptor_csv=descriptor_csv,
        molecules=prepared,
        common_parallel_scale=common_parallel_scale,
        output_files=output_files,
    )

    print("=" * 78)
    print("[OK] Figure 7 written:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")
    print(f"  {out_tiff}")
    print(f"  {out_summary}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)