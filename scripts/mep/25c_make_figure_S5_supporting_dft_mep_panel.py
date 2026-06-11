# -*- coding: utf-8 -*-
"""
25c_make_figure_S5_supporting_dft_mep_panel.py

Final polished Supporting Figure S5 renderer for the six auxiliary DFT/MEP compounds.

Panels:
A: CMR_GOLD_055 — FM0 N-free serviceable reference
B: CMR_GOLD_079 — FM3 mixed high-N regime
C: CMR_GOLD_016 — FM4 N-free pi-extended boundary
D: CMR_GOLD_090 — FM2 class-central anchor
E: CMR_GOLD_020 — FM2 N=3 high-Spread4 conjugated anchor
F: CMR_GOLD_092 — FM2 severe high-risk / high-disagreement anchor

This script reuses the tested Figure 7 rendering functions from:
scripts/24e_make_figure7_four_panel_from_cubes.py

Final protocol:
- electron-density isosurface: 0.004 a.u.
- common ESP scale: -0.08 to +0.08 a.u.
- white background
- thinner Q1-style annotation boxes
- slightly increased visual clearance for elongated / partially obscured structures

Outputs:
figures/supplementary/Figure_S5_supporting_DFT_MEP_panel.png
figures/supplementary/Figure_S5_supporting_DFT_MEP_panel.pdf
figures/supplementary/Figure_S5_supporting_DFT_MEP_panel.tiff
figures/supplementary/Figure_S5_supporting_DFT_MEP_panel_summary.txt

Run:
cd /d D:\\Makaleler\\coumarin-logp-working-source

python scripts\\25c_make_figure_S5_supporting_dft_mep_panel.py ^
  --project-root "D:\\Makaleler\\coumarin-logp-working-source" ^
  --descriptor-csv "D:\\Makaleler\\coumarin-logp-working-source\\data\\processed\\FigureS5_supporting_mep_descriptors.csv"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


TARGET_ORDER = [
    "CMR_GOLD_055",
    "CMR_GOLD_079",
    "CMR_GOLD_016",
    "CMR_GOLD_090",
    "CMR_GOLD_020",
    "CMR_GOLD_092",
]


DEFAULT_DESCRIPTOR_ROWS = [
    {
        "compound_id": "CMR_GOLD_055",
        "panel_label": "A",
        "role_label": "FM0 N-free reference",
        "n_count": 0,
        "delta_logp_consensus": -0.4300,
        "dipole_D": 6.5831,
        "gap_eV": 4.6371,
        "homo_eV": -6.5758,
        "lumo_eV": -1.9387,
    },
    {
        "compound_id": "CMR_GOLD_079",
        "panel_label": "B",
        "role_label": "FM3 mixed high-N case",
        "n_count": 4,
        "delta_logp_consensus": 0.0500,
        "dipole_D": 7.4431,
        "gap_eV": 3.8568,
        "homo_eV": -6.1912,
        "lumo_eV": -2.3344,
    },
    {
        "compound_id": "CMR_GOLD_016",
        "panel_label": "C",
        "role_label": "FM4 pi-extended boundary",
        "n_count": 0,
        "delta_logp_consensus": 2.2700,
        "dipole_D": 3.4024,
        "gap_eV": 4.4993,
        "homo_eV": -6.3417,
        "lumo_eV": -1.8424,
    },
    {
        "compound_id": "CMR_GOLD_090",
        "panel_label": "D",
        "role_label": "FM2 class-central anchor",
        "n_count": 1,
        "delta_logp_consensus": -0.8480,
        "dipole_D": 5.8133,
        "gap_eV": 4.1552,
        "homo_eV": -5.9040,
        "lumo_eV": -1.7488,
    },
    {
        "compound_id": "CMR_GOLD_020",
        "panel_label": "E",
        "role_label": "FM2 N=3 conjugated anchor",
        "n_count": 3,
        "delta_logp_consensus": -1.2730,
        "dipole_D": 2.7856,
        "gap_eV": 3.5635,
        "homo_eV": -5.8578,
        "lumo_eV": -2.2943,
    },
    {
        "compound_id": "CMR_GOLD_092",
        "panel_label": "F",
        "role_label": "FM2 severe high-risk anchor",
        "n_count": 1,
        "delta_logp_consensus": -2.1510,
        "dipole_D": 5.1942,
        "gap_eV": 3.9417,
        "homo_eV": -5.9005,
        "lumo_eV": -1.9588,
    },
]


# -------------------------------------------------------------------------
# Q1-style final visual tuning
# -------------------------------------------------------------------------

# Additional camera scaling:
# Larger value = molecule appears slightly smaller with more white margin.
# Smaller value = molecule appears larger.
# These panel-specific values are only for visual fit and do not affect ESP values.
PANEL_CAMERA_SCALE_MULTIPLIER = {
    "CMR_GOLD_055": 0.88,  # small molecule; enlarge slightly
    "CMR_GOLD_079": 1.04,  # elongated; add mild margin
    "CMR_GOLD_016": 1.13,  # branched/extended; add margin
    "CMR_GOLD_090": 1.10,  # elongated; add margin
    "CMR_GOLD_020": 1.03,  # broad but readable
    "CMR_GOLD_092": 1.08,  # crowded 3D structure; add margin
}

# Annotation y-position inside each panel.
# Smaller value = lower annotation box.
PANEL_ANNOTATION_Y = {
    "CMR_GOLD_055": 0.040,
    "CMR_GOLD_079": 0.034,
    "CMR_GOLD_016": 0.020,
    "CMR_GOLD_090": 0.025,
    "CMR_GOLD_020": 0.018,  # one step lower
    "CMR_GOLD_092": 0.018,  # one step lower
}

# Overall base margin added to the computed common scale.
BASE_CAMERA_MARGIN_MULTIPLIER = 1.035

# Annotation style.
ANNOTATION_FONTSIZE = 8.2
ANNOTATION_LINEWIDTH = 0.55
ANNOTATION_ALPHA = 0.88
ANNOTATION_PAD = 0.23

# Surface / ball-stick visibility tuning.
SURFACE_OPACITY_FINAL = 0.56
ATOM_OPACITY_FINAL = 0.98
BOND_OPACITY_FINAL = 0.92

# Final canvas.
FIG_WIDTH = 16.2
FIG_HEIGHT = 10.8
FINAL_DPI = 600


def load_fig7_module(project_root: Path):
    module_path = project_root / "scripts" / "24e_make_figure7_four_panel_from_cubes.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Figure 7 module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("fig7_renderer", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fig7_renderer"] = mod
    spec.loader.exec_module(mod)

    # Enforce final manuscript/SI protocol.
    mod.SURFACE_ISOVALUE = 0.004
    mod.GLOBAL_CLIP = 0.080

    # Polished SI rendering.
    mod.SURFACE_OPACITY = SURFACE_OPACITY_FINAL
    mod.ATOM_OPACITY = ATOM_OPACITY_FINAL
    mod.BOND_OPACITY = BOND_OPACITY_FINAL

    # Slightly improve internal structure visibility.
    mod.ATOM_RADIUS_HEAVY = 0.355
    mod.ATOM_RADIUS_H = 0.225
    mod.BOND_RADIUS_HEAVY = 0.108
    mod.BOND_RADIUS_H = 0.080

    # Render at slightly higher panel resolution.
    mod.SCREENSHOT_SIZE = (2450, 1750)

    return mod


def ensure_descriptor_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    df = pd.DataFrame(DEFAULT_DESCRIPTOR_ROWS)
    df.to_csv(path, index=False)
    print(f"[INFO] Figure S5 descriptor CSV created: {path}")


def load_descriptors(path: Path) -> pd.DataFrame:
    ensure_descriptor_csv(path)
    df = pd.read_csv(path)

    required = {
        "compound_id",
        "panel_label",
        "role_label",
        "n_count",
        "delta_logp_consensus",
        "dipole_D",
        "gap_eV",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Descriptor CSV missing columns: {sorted(missing)}")

    df = df[df["compound_id"].isin(TARGET_ORDER)].copy()

    if len(df) != 6:
        present = set(df["compound_id"].tolist())
        absent = [cid for cid in TARGET_ORDER if cid not in present]
        raise ValueError(f"Descriptor CSV missing Figure S5 compounds: {absent}")

    order_map = {cid: i for i, cid in enumerate(TARGET_ORDER)}
    df["__order__"] = df["compound_id"].map(order_map)
    df = df.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)

    return df


def fmt_float(value, digits=2, signed=False):
    try:
        if pd.isna(value):
            return None
        x = float(value)
    except Exception:
        return None

    if signed:
        return f"{x:+.{digits}f}"

    return f"{x:.{digits}f}"


def fmt_int(value):
    try:
        if pd.isna(value):
            return None
        return str(int(round(float(value))))
    except Exception:
        return None


def build_s5_annotation_text(mol: dict) -> str:
    role = str(mol.get("role_label", ""))
    n_text = fmt_int(mol.get("n_count", np.nan))
    dlogp = fmt_float(mol.get("delta_logp_consensus", np.nan), digits=2, signed=True)
    dipole = fmt_float(mol.get("dipole_D", np.nan), digits=2)
    gap = fmt_float(mol.get("gap_eV", np.nan), digits=2)

    lines = []

    if role:
        lines.append(role)

    line2 = []

    if n_text is not None:
        line2.append(f"N = {n_text}")

    if dlogp is not None:
        line2.append(r"$\Delta$logP$_{cons}$ = " + dlogp)

    if line2:
        lines.append(" | ".join(line2))

    line3 = []

    if dipole is not None:
        line3.append(r"$\mu$ = " + f"{dipole} D")

    if gap is not None:
        line3.append(f"Gap = {gap} eV")

    if line3:
        lines.append(" | ".join(line3))

    return "\n".join(lines)


def render_panel_image_s5(fig7, mol: dict, base_common_parallel_scale: float) -> np.ndarray:
    """
    Custom Figure S5 renderer.

    This keeps the same ESP scale and density isosurface as Figure 7, but allows
    gentle per-panel camera-margin tuning so elongated or crowded molecules are
    not visually cramped.
    """
    cid = mol["compound_id"]
    scale_multiplier = PANEL_CAMERA_SCALE_MULTIPLIER.get(cid, 1.0)
    panel_scale = base_common_parallel_scale * scale_multiplier

    plotter = fig7.setup_plotter()
    plotter.set_background("white")

    try:
        plotter.disable_depth_peeling()
    except Exception:
        pass

    plotter.add_mesh(
        mol["surface"],
        scalars="esp",
        cmap=fig7.MEP_CMAP,
        clim=[-fig7.GLOBAL_CLIP, fig7.GLOBAL_CLIP],
        opacity=fig7.SURFACE_OPACITY,
        show_edges=False,
        smooth_shading=True,
        ambient=fig7.SURFACE_AMBIENT,
        diffuse=fig7.SURFACE_DIFFUSE,
        specular=fig7.SURFACE_SPECULAR,
        specular_power=fig7.SURFACE_SPECULAR_POWER,
        lighting=True,
        show_scalar_bar=False,
    )

    try:
        plotter.remove_scalar_bar()
    except Exception:
        pass

    fig7.add_internal_molecule(plotter, mol["elements"], mol["coords"], mol["bonds"])
    fig7.apply_camera(plotter, mol["bounds"], panel_scale)

    plotter.render()
    img = plotter.screenshot(return_img=True, transparent_background=False)
    plotter.close()

    return np.asarray(img)


def assemble_s5_figure(fig7, panel_images, molecules, out_png, out_pdf, out_tiff):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FINAL_DPI, facecolor="white")

    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        height_ratios=[1.0, 1.0],
        wspace=0.025,
        hspace=0.070,
        left=0.020,
        right=0.970,
        top=0.955,
        bottom=0.047,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    cax = fig.add_subplot(gs[:, 3])

    for ax, img, mol in zip(axes, panel_images, molecules):
        cid = mol["compound_id"]
        ax.imshow(img)
        ax.axis("off")
        ax.set_facecolor("white")

        # Panel label.
        ax.text(
            0.018,
            0.965,
            mol["panel_label"],
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

        # Compound ID.
        ax.text(
            0.50,
            0.990,
            mol["title"],
            transform=ax.transAxes,
            fontsize=14.5,
            fontweight="bold",
            va="top",
            ha="center",
            color="black",
        )

        annotation = build_s5_annotation_text(mol)
        annotation_y = PANEL_ANNOTATION_Y.get(cid, 0.030)

        ax.text(
            0.50,
            annotation_y,
            annotation,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONTSIZE,
            va="bottom",
            ha="center",
            color="black",
            linespacing=1.10,
            bbox=dict(
                boxstyle=f"square,pad={ANNOTATION_PAD}",
                facecolor="white",
                edgecolor="#8A8A8A",
                linewidth=ANNOTATION_LINEWIDTH,
                alpha=ANNOTATION_ALPHA,
            ),
            zorder=20,
        )

    cax.set_facecolor("white")

    norm = Normalize(vmin=-fig7.GLOBAL_CLIP, vmax=fig7.GLOBAL_CLIP)
    sm = ScalarMappable(norm=norm, cmap=fig7.MEP_CMAP)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Electrostatic potential (a.u.)", fontsize=13, labelpad=11)
    cb.ax.tick_params(labelsize=9.5, width=1.0, length=4.5)
    cb.set_ticks([-0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08])

    for spine in cb.ax.spines.values():
        spine.set_linewidth(0.90)

    fig.patch.set_facecolor("white")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=FINAL_DPI, facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_pdf, dpi=FINAL_DPI, facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        out_tiff,
        dpi=FINAL_DPI,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
        pad_inches=0.025,
        pil_kwargs={"compression": "tiff_lzw"},
    )

    plt.close(fig)


def write_summary(fig7, out_summary, project_root, descriptor_csv, molecules, base_common_parallel_scale, output_files):
    lines = []
    lines.append("Figure S5 supporting DFT/MEP rendering summary")
    lines.append("=" * 78)
    lines.append(f"Project root: {project_root}")
    lines.append(f"Descriptor CSV: {descriptor_csv}")
    lines.append("")
    lines.append(f"Surface isovalue: {fig7.SURFACE_ISOVALUE:.6f} a.u.")
    lines.append(f"Common ESP scale: -{fig7.GLOBAL_CLIP:.3f} to +{fig7.GLOBAL_CLIP:.3f} a.u.")
    lines.append(f"ESP smoothing sigma: {fig7.ESP_SMOOTH_SIGMA:.3f}")
    lines.append(f"Base common camera parallel scale: {base_common_parallel_scale:.6f}")
    lines.append(f"Surface opacity: {fig7.SURFACE_OPACITY:.3f}")
    lines.append(f"Atom opacity: {fig7.ATOM_OPACITY:.3f}")
    lines.append(f"Bond opacity: {fig7.BOND_OPACITY:.3f}")
    lines.append("")
    lines.append("Outputs:")

    for label, path in output_files.items():
        lines.append(f"  {label}: {path}")

    lines.append("")
    lines.append("Panel files and ESP surface statistics:")
    lines.append("-" * 78)

    for mol in molecules:
        cid = mol["compound_id"]
        vals = mol["esp_surface_values"]
        panel_scale = base_common_parallel_scale * PANEL_CAMERA_SCALE_MULTIPLIER.get(cid, 1.0)

        lines.append(f"{mol['panel_label']} | {mol['title']}")
        lines.append(f"  role label   : {mol.get('role_label', '')}")
        lines.append(f"  structure    : {mol['structure']}")
        lines.append(f"  density cube : {mol['density_cube']}")
        lines.append(f"  esp cube     : {mol['esp_cube']}")
        lines.append(f"  unit factor  : {mol['unit_factor']:.8f}")
        lines.append(f"  atoms        : {len(mol['elements'])}")
        lines.append(f"  bonds        : {len(mol['bonds'])}")
        lines.append(f"  panel camera scale multiplier: {PANEL_CAMERA_SCALE_MULTIPLIER.get(cid, 1.0):.3f}")
        lines.append(f"  panel camera parallel scale  : {panel_scale:.6f}")
        lines.append(f"  annotation y : {PANEL_ANNOTATION_Y.get(cid, 0.030):.3f}")
        lines.append(f"  min ESP      : {np.min(vals):.6f}")
        lines.append(f"  max ESP      : {np.max(vals):.6f}")
        lines.append(f"  mean ESP     : {np.mean(vals):.6f}")
        lines.append("")

    out_summary.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--descriptor-csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    descriptor_csv = Path(args.descriptor_csv)

    fig7 = load_fig7_module(project_root)
    df = load_descriptors(descriptor_csv)

    print("=" * 78)
    print("Preparing polished Figure S5 supporting DFT/MEP panel")
    print("=" * 78)
    print(f"Project root: {project_root}")
    print(f"Descriptor CSV: {descriptor_csv}")
    print(f"Surface isovalue: {fig7.SURFACE_ISOVALUE:.6f} a.u.")
    print(f"Common ESP scale: -{fig7.GLOBAL_CLIP:.3f} to +{fig7.GLOBAL_CLIP:.3f} a.u.")
    print("-" * 78)

    prepared = []

    for _, row in df.iterrows():
        cid = row["compound_id"]
        print(f"Preparing {cid}...")
        mol = fig7.prepare_molecule(row, project_root)
        vals = mol["esp_surface_values"]
        print(f"  ESP stats: min={np.min(vals):.5f}, max={np.max(vals):.5f}, mean={np.mean(vals):.5f}")
        prepared.append(mol)

    base_common_parallel_scale = fig7.compute_common_camera_scale(prepared) * BASE_CAMERA_MARGIN_MULTIPLIER

    print("-" * 78)
    print(f"Base common camera parallel scale: {base_common_parallel_scale:.6f}")
    print("-" * 78)

    panel_images = []

    for mol in prepared:
        cid = mol["compound_id"]
        scale_mult = PANEL_CAMERA_SCALE_MULTIPLIER.get(cid, 1.0)
        print(f"Rendering {mol['title']} with scale multiplier {scale_mult:.3f}...")
        img = render_panel_image_s5(fig7, mol, base_common_parallel_scale)
        panel_images.append(img)

    out_dir = project_root / "figures" / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "Figure_S5_supporting_DFT_MEP_panel.png"
    out_pdf = out_dir / "Figure_S5_supporting_DFT_MEP_panel.pdf"
    out_tiff = out_dir / "Figure_S5_supporting_DFT_MEP_panel.tiff"
    out_summary = out_dir / "Figure_S5_supporting_DFT_MEP_panel_summary.txt"

    print("Assembling polished Figure S5...")
    assemble_s5_figure(fig7, panel_images, prepared, out_png, out_pdf, out_tiff)

    output_files = {
        "PNG": out_png,
        "PDF": out_pdf,
        "TIFF": out_tiff,
        "SUMMARY": out_summary,
    }

    write_summary(
        fig7=fig7,
        out_summary=out_summary,
        project_root=project_root,
        descriptor_csv=descriptor_csv,
        molecules=prepared,
        base_common_parallel_scale=base_common_parallel_scale,
        output_files=output_files,
    )

    print("=" * 78)
    print("[OK] Polished Figure S5 written:")
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