# -*- coding: utf-8 -*-
"""
Final Figure 7 MEP rendering script
Uses corrected CMR_GOLD_043 and corrected CMR_GOLD_058 files.

Run:
    conda activate mepfig
    cd /d "C:\\orca_tests\\DFT_FINAL_PROPERTIES\\058\\04_figures"
    python make_mep_figure_final_correct043_correct058.py

Required files in the working directory:
    CMR_GOLD_043_sp_charge_clean.xyz
    CMR_GOLD_043_sp_charge_clean.eldens.cube
    CMR_GOLD_043_sp_charge_clean.scfp.esp.cube

    CMR_GOLD_058_correct_opt_pal8.xyz
    CMR_GOLD_058_sp_charge_clean.eldens.cube
    CMR_GOLD_058_sp_charge_clean.scfp.esp.cube

Outputs:
    Figure7_MEP_final_correct043_correct058.png
    Figure7_MEP_final_correct043_correct058.tiff
    Figure7_MEP_final_correct043_correct058_summary.txt
"""

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# USER SETTINGS
# ============================================================

SURFACE_ISOVALUE = 0.001
GLOBAL_CLIP = 0.080
ESP_SMOOTH_SIGMA = 0.80

OUTPUT_PNG = "Figure7_MEP_final_correct043_correct058.png"
OUTPUT_TIFF = "Figure7_MEP_final_correct043_correct058.tiff"
OUTPUT_SUMMARY = "Figure7_MEP_final_correct043_correct058_summary.txt"
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

CAMERA_MARGIN = 1.03
CAMERA_Z_FACTOR = 2.45
CAMERA_X_TILT = 0.14
CAMERA_Y_TILT = -0.08

BOHR_TO_ANG = 0.52917721092

MOLECULES = [
    {
        "panel_label": "A",
        "title": "CMR-GOLD-043",
        "structure": "CMR_GOLD_043_sp_charge_clean.xyz",
        "density_cube": "CMR_GOLD_043_sp_charge_clean.eldens.cube",
        "esp_cube": "CMR_GOLD_043_sp_charge_clean.scfp.esp.cube",
        "expected_formula": "C11H6N2O3",
        "expected_components": 1,
    },
    {
        "panel_label": "B",
        "title": "CMR-GOLD-058",
        "structure": "CMR_GOLD_058_correct_opt_pal8.xyz",
        "density_cube": "CMR_GOLD_058_sp_charge_clean.eldens.cube",
        "esp_cube": "CMR_GOLD_058_sp_charge_clean.scfp.esp.cube",
        "expected_components": 1,
    },
]

# Negative ESP = red, positive ESP = blue.
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
# FILE CHECKS
# ============================================================

def check_files():
    missing = []

    for mol in MOLECULES:
        for key in ("structure", "density_cube", "esp_cube"):
            if not os.path.exists(mol[key]):
                missing.append(mol[key])

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  - " + "\n  - ".join(missing)
        )


# ============================================================
# STRUCTURE READERS
# ============================================================

def safe_symbol(text):
    if not text:
        return "C"

    t = text.strip()

    if not t:
        return "C"

    if len(t) >= 2 and t[:2].capitalize() in ("Cl", "Br"):
        return t[:2].capitalize()

    return t[0].upper()


def read_xyz(xyz_path):
    with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

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
            "XYZ atom count mismatch in {}. Expected {}, got {}.".format(
                xyz_path,
                natoms,
                len(coords),
            )
        )

    return np.array(elements, dtype=object), np.array(coords, dtype=float)


def read_pdb(pdb_path):
    elements = []
    coords = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except Exception:
                    continue

                el = line[76:78].strip()

                if not el:
                    name = line[12:16].strip()
                    letters = "".join([c for c in name if c.isalpha()])
                    el = safe_symbol(letters)
                else:
                    el = safe_symbol(el)

                elements.append(el)
                coords.append([x, y, z])

    if len(coords) == 0:
        raise ValueError("No atoms could be read from PDB: {}".format(pdb_path))

    return np.array(elements, dtype=object), np.array(coords, dtype=float)


def read_structure(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".xyz":
        return read_xyz(path)

    if ext == ".pdb":
        return read_pdb(path)

    raise ValueError("Unsupported structure file format: {}".format(path))


def formula_from_elements(elements):
    order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    counts = {}

    for e in elements:
        counts[e] = counts.get(e, 0) + 1

    parts = []

    for e in order:
        if e in counts:
            parts.append("{}{}".format(e, counts[e] if counts[e] > 1 else ""))

    for e in sorted(counts):
        if e not in order:
            parts.append("{}{}".format(e, counts[e] if counts[e] > 1 else ""))

    return "".join(parts)


# ============================================================
# CUBE READER
# ============================================================

def read_cube(cube_path):
    with open(cube_path, "r", encoding="utf-8", errors="ignore") as f:
        _comment1 = f.readline()
        _comment2 = f.readline()

        line = f.readline().split()
        natoms = abs(int(float(line[0])))
        origin = np.array(
            [float(line[1]), float(line[2]), float(line[3])],
            dtype=float,
        )

        dims = []
        axes = []

        for _ in range(3):
            parts = f.readline().split()
            n = abs(int(float(parts[0])))
            vec = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=float,
            )
            dims.append(n)
            axes.append(vec)

        atom_numbers = []
        atom_coords = []

        for _ in range(natoms):
            parts = f.readline().split()
            anum = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            atom_numbers.append(anum)
            atom_coords.append([x, y, z])

        values = []

        for line in f:
            if line.strip():
                values.extend(line.split())

    dims = tuple(dims)
    values = np.array(values, dtype=np.float32)
    expected = dims[0] * dims[1] * dims[2]

    if values.size != expected:
        raise ValueError(
            "{}: cube data size mismatch. Expected {}, got {}".format(
                cube_path,
                expected,
                values.size,
            )
        )

    data = values.reshape(dims, order="C")

    return {
        "origin": origin,
        "axes": np.array(axes, dtype=float),
        "dims": np.array(dims, dtype=int),
        "data": data,
        "atom_numbers": np.array(atom_numbers, dtype=int),
        "atom_coords": np.array(atom_coords, dtype=float),
    }


# ============================================================
# UNIT HARMONIZATION
# ============================================================

def guess_cube_to_angstrom_factor(cube_atom_coords, structure_coords):
    n = min(len(cube_atom_coords), len(structure_coords))

    if n < 2:
        return 1.0

    cube_distances = []
    struct_distances = []

    for i in range(n):
        for j in range(i + 1, n):
            dc = np.linalg.norm(cube_atom_coords[i] - cube_atom_coords[j])
            ds = np.linalg.norm(structure_coords[i] - structure_coords[j])

            if ds > 1e-6:
                cube_distances.append(dc)
                struct_distances.append(ds)

    if len(cube_distances) == 0:
        return 1.0

    ratio = np.median(np.array(cube_distances) / np.array(struct_distances))

    if 1.65 < ratio < 2.10:
        return BOHR_TO_ANG

    if 0.85 < ratio < 1.15:
        return 1.0

    return 1.0


def scale_cube_to_angstrom(cube_obj, factor):
    cube_obj = dict(cube_obj)
    cube_obj["origin"] = cube_obj["origin"] * factor
    cube_obj["axes"] = cube_obj["axes"] * factor
    cube_obj["atom_coords"] = cube_obj["atom_coords"] * factor
    return cube_obj


# ============================================================
# GRID / SURFACE
# ============================================================

def cube_to_imagedata(data3d, origin, axes, scalar_name):
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


def safe_compute_normals(poly):
    try:
        return poly.compute_normals(
            auto_orient_normals=True,
            consistent_normals=True,
            inplace=False,
        )
    except TypeError:
        try:
            return poly.compute_normals(
                auto_orient_normals=True,
                inplace=False,
            )
        except TypeError:
            return poly.compute_normals(inplace=False)


def smooth_surface(poly):
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


def build_density_surface(density_cube, isovalue):
    grid = cube_to_imagedata(
        density_cube["data"],
        density_cube["origin"],
        density_cube["axes"],
        "density",
    )

    surface = grid.contour(isosurfaces=[isovalue], scalars="density")

    if surface.n_points == 0:
        raise ValueError("No surface generated at isovalue {}.".format(isovalue))

    surface = surface.triangulate()
    surface = surface.clean()
    surface = smooth_surface(surface)
    surface = safe_compute_normals(surface)
    return surface


def build_adjacency(mesh):
    neighbors = [set() for _ in range(mesh.n_points)]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    return neighbors


def smooth_scalars_on_surface(mesh, values, iterations=4, alpha=0.45):
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


def sample_esp_onto_surface(surface, esp_cube, smooth_iterations=4):
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
# BONDS / CONNECTIVITY / MOLECULE RENDERING
# ============================================================

def infer_bonds(elements, coords):
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


def connected_components(n, bonds):
    adjacency = [[] for _ in range(n)]

    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

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

            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        comps.append(comp)

    comps.sort(key=len, reverse=True)
    return comps


def validate_connectivity(title, elements, coords, bonds, expected_components=1):
    comps = connected_components(len(elements), bonds)

    if len(comps) != expected_components:
        labels = []

        for k, comp in enumerate(comps, 1):
            atom_labels = ["{}{}".format(elements[i], i + 1) for i in comp]
            labels.append("component {}: {}".format(k, ", ".join(atom_labels)))

        raise RuntimeError(
            "{} connectivity failed. Expected {} component(s), got {}.\n{}".format(
                title,
                expected_components,
                len(comps),
                "\n".join(labels),
            )
        )

    return comps


def get_atom_color(sym):
    return ATOM_COLORS.get(sym, "#B8B8B8")


def get_atom_radius(sym):
    if sym == "H":
        return ATOM_RADIUS_H

    return ATOM_RADIUS_HEAVY


def get_bond_radius(sym1, sym2):
    if sym1 == "H" or sym2 == "H":
        return BOND_RADIUS_H

    return BOND_RADIUS_HEAVY


def add_internal_molecule(plotter, elements, coords, bonds):
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


# ============================================================
# PLOTTER / CAMERA
# ============================================================

def setup_plotter():
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


def combined_bounds(surface, coords):
    combined = np.vstack([surface.points, coords])

    return (
        np.min(combined[:, 0]),
        np.max(combined[:, 0]),
        np.min(combined[:, 1]),
        np.max(combined[:, 1]),
        np.min(combined[:, 2]),
        np.max(combined[:, 2]),
    )


def bounds_center(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    return np.array(
        [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ],
        dtype=float,
    )


def bounds_spans(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    return np.array(
        [
            xmax - xmin,
            ymax - ymin,
            zmax - zmin,
        ],
        dtype=float,
    )


def compute_common_camera_scale(prepared):
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


def apply_camera(plotter, bounds, common_parallel_scale):
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


def render_panel_image(mol_data, common_parallel_scale):
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
    img = plotter.screenshot(return_img=True, transparent_background=False)
    plotter.close()

    return np.asarray(img)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_molecule(cfg):
    elements, coords = read_structure(cfg["structure"])

    formula = formula_from_elements(elements)

    if "expected_formula" in cfg and formula != cfg["expected_formula"]:
        raise RuntimeError(
            "{} formula mismatch. Expected {}, got {}.".format(
                cfg["title"],
                cfg["expected_formula"],
                formula,
            )
        )

    dens_cube = read_cube(cfg["density_cube"])
    esp_cube = read_cube(cfg["esp_cube"])

    factor = guess_cube_to_angstrom_factor(dens_cube["atom_coords"], coords)

    dens_cube = scale_cube_to_angstrom(dens_cube, factor)
    esp_cube = scale_cube_to_angstrom(esp_cube, factor)

    surface = build_density_surface(dens_cube, SURFACE_ISOVALUE)

    smooth_iter = 4

    if "058" in cfg["title"]:
        smooth_iter = 7

    surface, esp_surface = sample_esp_onto_surface(
        surface,
        esp_cube,
        smooth_iterations=smooth_iter,
    )

    bonds = infer_bonds(elements, coords)
    comps = validate_connectivity(
        cfg["title"],
        elements,
        coords,
        bonds,
        expected_components=cfg.get("expected_components", 1),
    )

    bnds = combined_bounds(surface, coords)

    return {
        "panel_label": cfg["panel_label"],
        "title": cfg["title"],
        "elements": elements,
        "coords": coords,
        "bonds": bonds,
        "components": comps,
        "formula": formula,
        "surface": surface,
        "esp_surface_values": esp_surface,
        "bounds": bnds,
        "structure": cfg["structure"],
        "density_cube": cfg["density_cube"],
        "esp_cube": cfg["esp_cube"],
    }


# ============================================================
# FINAL FIGURE ASSEMBLY
# ============================================================

def assemble_final_figure(panel_images, molecules, output_png, output_tiff):
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FINAL_DPI)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.00, 1.00, 0.052],
        wspace=0.030,
        left=0.020,
        right=0.965,
        top=0.925,
        bottom=0.070,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    axes = [ax1, ax2]

    for ax, img, mol in zip(axes, panel_images, molecules):
        ax.imshow(img)
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
    sm = ScalarMappable(norm=norm, cmap=MEP_CMAP)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Electrostatic potential (a.u.)", fontsize=16, labelpad=14)
    cb.ax.tick_params(labelsize=12, width=1.2, length=5)
    cb.set_ticks(
        [-0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08]
    )

    for spine in cb.ax.spines.values():
        spine.set_linewidth(1.1)

    fig.patch.set_facecolor("white")

    fig.savefig(
        output_png,
        dpi=FINAL_DPI,
        facecolor="white",
        bbox_inches="tight",
    )

    fig.savefig(
        output_tiff,
        dpi=FINAL_DPI,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)


def write_summary(molecules, common_parallel_scale):
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("Final Figure 7 MEP rendering summary\n")
        f.write("=" * 72 + "\n")
        f.write("Surface isovalue: {:.6f} a.u.\n".format(SURFACE_ISOVALUE))
        f.write("Common ESP scale: -{:.3f} to +{:.3f} a.u.\n".format(GLOBAL_CLIP, GLOBAL_CLIP))
        f.write("Common camera parallel scale: {:.6f}\n".format(common_parallel_scale))
        f.write("Output PNG: {}\n".format(OUTPUT_PNG))
        f.write("Output TIFF: {}\n".format(OUTPUT_TIFF))
        f.write("\n")

        for mol in molecules:
            vals = mol["esp_surface_values"]
            f.write("{}\n".format(mol["title"]))
            f.write("  formula      : {}\n".format(mol["formula"]))
            f.write("  components   : {}\n".format(len(mol["components"])))
            f.write("  structure    : {}\n".format(mol["structure"]))
            f.write("  density cube : {}\n".format(mol["density_cube"]))
            f.write("  esp cube     : {}\n".format(mol["esp_cube"]))
            f.write("  min ESP      : {:.6f}\n".format(np.min(vals)))
            f.write("  max ESP      : {:.6f}\n".format(np.max(vals)))
            f.write("  mean ESP     : {:.6f}\n".format(np.mean(vals)))
            f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Preparing final Figure 7 with corrected CMR-GOLD-043 and CMR-GOLD-058")
    print("=" * 72)

    check_files()

    prepared = []

    for cfg in MOLECULES:
        print("Reading and preparing:", cfg["title"])
        print("  structure:", cfg["structure"])
        print("  density  :", cfg["density_cube"])
        print("  esp      :", cfg["esp_cube"])

        mol = prepare_molecule(cfg)
        prepared.append(mol)

        vals = mol["esp_surface_values"]

        print("  Formula:", mol["formula"])
        print("  Connected components:", len(mol["components"]))
        print(
            "  Surface ESP stats for {}: min={:.5f}, max={:.5f}, mean={:.5f}".format(
                cfg["title"],
                np.min(vals),
                np.max(vals),
                np.mean(vals),
            )
        )

    common_parallel_scale = compute_common_camera_scale(prepared)

    print("-" * 72)
    print("Common fixed ESP scale: +/- {:.5f} a.u.".format(GLOBAL_CLIP))
    print("Common camera parallel scale: {:.5f}".format(common_parallel_scale))
    print("Both panels rendered with the same physical scale.")
    print("-" * 72)

    panel_images = []

    for mol in prepared:
        print("Rendering panel:", mol["title"])
        img = render_panel_image(mol, common_parallel_scale)
        panel_images.append(img)

    print("Assembling final PNG and TIFF...")
    assemble_final_figure(panel_images, prepared, OUTPUT_PNG, OUTPUT_TIFF)
    write_summary(prepared, common_parallel_scale)

    print("=" * 72)
    print("Done.")
    print("Output written to:")
    print("  {}".format(OUTPUT_PNG))
    print("  {}".format(OUTPUT_TIFF))
    print("  {}".format(OUTPUT_SUMMARY))
    print("=" * 72)


if __name__ == "__main__":
    main()