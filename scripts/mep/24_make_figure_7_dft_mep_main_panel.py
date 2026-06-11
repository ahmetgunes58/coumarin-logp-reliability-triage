# -*- coding: utf-8 -*-
"""
24_make_figure_7_dft_mep_main_panel.py

Generate Figure 7:
Targeted DFT/MEP diagnostic comparison of matched N = 2 coumarins.

Main-panel compounds:
A: CMR_GOLD_043 — compact N = 2 serviceable control
B: CMR_GOLD_044 — compact N = 2 serviceable control
C: CMR_GOLD_029 — polar N = 2 overestimation / high-dipole case
D: CMR_GOLD_058 — N = 2 donor-acceptor-conjugated severe overestimation case

Outputs:
- Figure_7_DFT_MEP_main_panel.png
- Figure_7_DFT_MEP_main_panel.pdf
- Figure_7_DFT_MEP_main_panel.tiff
- Figure_7_DFT_MEP_main_panel_manifest.txt

Recommended usage:
python scripts\\24_make_figure_7_dft_mep_main_panel.py ^
  --project-root "D:\\Makaleler\\coumarin-logp-working-source" ^
  --mep-dir "D:\\Makaleler\\coumarin-logp-working-source\\figures\\mep" ^
  --descriptor-csv "D:\\Makaleler\\coumarin-logp-working-source\\data\\processed\\Dataset_SXX_DFT_panel_10_descriptors.csv"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Figure definition
# ---------------------------------------------------------------------

COMPOUNDS = [
    "CMR_GOLD_043",
    "CMR_GOLD_044",
    "CMR_GOLD_029",
    "CMR_GOLD_058",
]

PANEL_LABELS = {
    "CMR_GOLD_043": "A",
    "CMR_GOLD_044": "B",
    "CMR_GOLD_029": "C",
    "CMR_GOLD_058": "D",
}

ROLE_LABELS = {
    "CMR_GOLD_043": "Compact N = 2 control",
    "CMR_GOLD_044": "Compact N = 2 control",
    "CMR_GOLD_029": "Polar overestimation",
    "CMR_GOLD_058": "D–A-conjugated severe case",
}

# Fallback values are taken from the current manuscript/SI context.
# Prefer a deposited descriptor CSV/XLSX for the final reproducible figure.
FALLBACK_DESCRIPTORS = {
    "CMR_GOLD_043": {
        "FM_label": "FM2",
        "N_count": 2,
        "logP_exp": 1.9500,
        "Consensus_logP": 1.78,
        "Delta_logP_consensus": +0.1700,
        "Dipole_D": 5.7832,
        "Gap_eV": 4.1157,
        "HOMO_eV": np.nan,
        "LUMO_eV": np.nan,
    },
    "CMR_GOLD_044": {
        "FM_label": "FM2",
        "N_count": 2,
        "logP_exp": 2.1200,
        "Consensus_logP": 2.15,
        "Delta_logP_consensus": -0.0300,
        "Dipole_D": 6.3502,
        "Gap_eV": 4.1451,
        "HOMO_eV": -6.6834,
        "LUMO_eV": -2.5383,
    },
    "CMR_GOLD_029": {
        "FM_label": "FM1",
        "N_count": 2,
        "logP_exp": 0.5500,
        "Consensus_logP": 2.59,
        "Delta_logP_consensus": -2.0400,
        "Dipole_D": 10.3695,
        "Gap_eV": 4.4523,
        "HOMO_eV": np.nan,
        "LUMO_eV": np.nan,
    },
    "CMR_GOLD_058": {
        "FM_label": "FM1",
        "N_count": 2,
        "logP_exp": 0.9664,
        "Consensus_logP": 6.16,
        "Delta_logP_consensus": -5.1936,
        "Dipole_D": 6.1104,
        "Gap_eV": 2.6080,
        "HOMO_eV": -4.9939,
        "LUMO_eV": -2.3859,
    },
}


# Optional: if automatic image search fails, paste exact image paths here.
# Example:
# EXPLICIT_IMAGE_FILES = {
#     "CMR_GOLD_043": r"D:\...\CMR_GOLD_043_MEP.png",
#     "CMR_GOLD_044": r"D:\...\CMR_GOLD_044_MEP.png",
#     "CMR_GOLD_029": r"D:\...\CMR_GOLD_029_MEP.png",
#     "CMR_GOLD_058": r"D:\...\CMR_GOLD_058_MEP.png",
# }
EXPLICIT_IMAGE_FILES: Dict[str, Optional[str]] = {
    "CMR_GOLD_043": None,
    "CMR_GOLD_044": None,
    "CMR_GOLD_029": None,
    "CMR_GOLD_058": None,
}


# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------

def normalise_name(text: str) -> str:
    """Normalise a column name for flexible matching."""
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


COLUMN_ALIASES = {
    "Compound_ID": [
        "Compound_ID", "compound_id", "Compound", "ID", "Molecule", "Molecule_ID"
    ],
    "FM_label": [
        "FM_label", "FM", "Failure_mode", "FailureMode", "FM class", "FM_class"
    ],
    "N_count": [
        "N_count", "Ncount", "N", "Nitrogen_count", "NitrogenCount"
    ],
    "logP_exp": [
        "logP_exp", "Experimental_logP", "exp_logP", "logP experimental", "logPexp"
    ],
    "Consensus_logP": [
        "Consensus_logP", "consensus_logP", "SwissADME_consensus",
        "logP_consensus", "Consensus"
    ],
    "Delta_logP_consensus": [
        "Delta_logP_consensus", "delta_logP_consensus", "ΔlogP_consensus",
        "Consensus_error", "Error_consensus", "delta_consensus",
        "logP_exp_minus_consensus", "Exp_minus_Pred_consensus"
    ],
    "Dipole_D": [
        "Dipole_D", "Dipole", "Dipole_moment_D", "DipoleMoment_D",
        "dipole_debye", "mu_D", "μ_D"
    ],
    "Gap_eV": [
        "Gap_eV", "HOMO_LUMO_gap_eV", "HL_gap_eV", "DeltaE_HL_eV",
        "HOMO-LUMO gap", "Egap_eV"
    ],
    "HOMO_eV": [
        "HOMO_eV", "E_HOMO_eV", "HOMO", "HOMO_energy_eV"
    ],
    "LUMO_eV": [
        "LUMO_eV", "E_LUMO_eV", "LUMO", "LUMO_energy_eV"
    ],
}


def find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    """Find a dataframe column by alias."""
    aliases = COLUMN_ALIASES.get(canonical_name, [canonical_name])
    norm_to_col = {normalise_name(c): c for c in df.columns}

    for alias in aliases:
        key = normalise_name(alias)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def read_descriptor_table(path: Path) -> pd.DataFrame:
    """Read CSV/XLSX descriptor table."""
    if not path.exists():
        raise FileNotFoundError(f"Descriptor table not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported descriptor table format: {path.suffix}")


def load_descriptors(descriptor_path: Optional[Path]) -> Tuple[Dict[str, dict], str]:
    """
    Load descriptors for the four compounds.
    If no descriptor table is supplied, use fallback values.
    """
    descriptors = {cid: dict(vals) for cid, vals in FALLBACK_DESCRIPTORS.items()}
    source = "fallback values embedded in script"

    if descriptor_path is None:
        return descriptors, source

    if not descriptor_path.exists():
        print(f"[WARNING] Descriptor table not found: {descriptor_path}")
        print("[WARNING] Using fallback descriptor values embedded in the script.")
        return descriptors, source

    df = read_descriptor_table(descriptor_path)
    source = str(descriptor_path)

    compound_col = find_column(df, "Compound_ID")
    if compound_col is None:
        print("[WARNING] Could not find Compound_ID column in descriptor table.")
        print("[WARNING] Using fallback descriptor values embedded in the script.")
        return descriptors, "fallback values embedded in script"

    # Standardise requested columns if present
    mapped_cols = {}
    for canonical in COLUMN_ALIASES:
        col = find_column(df, canonical)
        if col is not None:
            mapped_cols[canonical] = col

    df["_compound_match"] = df[compound_col].astype(str).str.strip()

    for cid in COMPOUNDS:
        rows = df[df["_compound_match"] == cid]
        if rows.empty:
            print(f"[WARNING] {cid} not found in descriptor table; fallback values retained.")
            continue

        row = rows.iloc[0]
        for canonical, actual_col in mapped_cols.items():
            if canonical == "Compound_ID":
                continue
            value = row[actual_col]
            if pd.isna(value):
                continue
            descriptors[cid][canonical] = value

    return descriptors, source


def candidate_descriptor_path(project_root: Path) -> Optional[Path]:
    """Try to identify a likely DFT descriptor file automatically."""
    likely_patterns = [
        "Dataset*S*DFT*descriptor*.csv",
        "Dataset*S*DFT*frontier*.csv",
        "Dataset*S*DFT*summary*.csv",
        "*DFT_panel*descriptor*.csv",
        "*DFT_panel*summary*.csv",
        "*frontier*dipole*.csv",
    ]

    search_dirs = [
        project_root / "data" / "processed",
        project_root / "data",
        project_root / "outputs",
        project_root / "dft",
    ]

    candidates = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in likely_patterns:
            candidates.extend(directory.rglob(pattern))

    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p).lower()))
    return candidates[0] if candidates else None


def trim_whitespace(img: Image.Image, tolerance: int = 12, margin: int = 12) -> Image.Image:
    """
    Trim excess white/transparent margins from an image.
    Keeps a small margin to avoid cutting rendered molecule edges.
    """
    img = img.convert("RGBA")

    # Alpha-based trim if transparency exists
    alpha = np.array(img.getchannel("A"))
    if np.any(alpha < 255):
        mask = alpha > tolerance
    else:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        diff = ImageChops.difference(img, bg).convert("L")
        mask = np.array(diff) > tolerance

    if not np.any(mask):
        return img

    ys, xs = np.where(mask)
    left = max(int(xs.min()) - margin, 0)
    right = min(int(xs.max()) + margin, img.size[0])
    top = max(int(ys.min()) - margin, 0)
    bottom = min(int(ys.max()) + margin, img.size[1])

    return img.crop((left, top, right, bottom))


def find_mep_image(compound_id: str, project_root: Path, mep_dir: Optional[Path]) -> Path:
    """Find MEP PNG/TIF image for a compound."""
    explicit = EXPLICIT_IMAGE_FILES.get(compound_id)
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit image path for {compound_id} does not exist: {p}")

    search_dirs = []
    if mep_dir is not None:
        search_dirs.append(mep_dir)

    search_dirs.extend([
        project_root / "figures",
        project_root / "figures" / "mep",
        project_root / "outputs",
        project_root / "outputs" / "mep",
        project_root / "dft",
        project_root / "data" / "processed",
    ])

    patterns = [
        f"*{compound_id}*MEP*.png",
        f"*{compound_id}*mep*.png",
        f"*{compound_id}*ESP*.png",
        f"*{compound_id}*esp*.png",
        f"*{compound_id}*.png",
        f"*{compound_id}*MEP*.tif",
        f"*{compound_id}*mep*.tif",
        f"*{compound_id}*.tif",
        f"*{compound_id}*.tiff",
    ]

    candidates = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            candidates.extend(directory.rglob(pattern))

    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p).lower()))

    if not candidates:
        raise FileNotFoundError(
            f"No MEP image found for {compound_id}. "
            f"Set EXPLICIT_IMAGE_FILES in the script or pass --mep-dir."
        )

    return candidates[0]


def fmt_float(value, digits: int = 2, signed: bool = False) -> str:
    """Format numeric value robustly."""
    try:
        x = float(value)
    except Exception:
        return "n/a"

    if np.isnan(x):
        return "n/a"

    sign = "+" if signed else ""
    return f"{x:{sign}.{digits}f}"


def descriptor_text(compound_id: str, descriptors: Dict[str, dict], show_homo_lumo: bool = False) -> str:
    """Create compact descriptor annotation for each panel."""
    d = descriptors[compound_id]

    fm = str(d.get("FM_label", ""))
    n_count = d.get("N_count", "n/a")
    delta = fmt_float(d.get("Delta_logP_consensus", np.nan), digits=2, signed=True)
    dipole = fmt_float(d.get("Dipole_D", np.nan), digits=2)
    gap = fmt_float(d.get("Gap_eV", np.nan), digits=2)

    line1 = f"{fm}; N = {n_count}; ΔlogPcons = {delta}"
    line2 = f"μ = {dipole} D; ΔEHL = {gap} eV"

    if show_homo_lumo:
        homo = fmt_float(d.get("HOMO_eV", np.nan), digits=2)
        lumo = fmt_float(d.get("LUMO_eV", np.nan), digits=2)
        line3 = f"HOMO = {homo} eV; LUMO = {lumo} eV"
        return f"{line1}\n{line2}\n{line3}"

    return f"{line1}\n{line2}"


def save_tiff(fig, out_tiff: Path, dpi: int) -> None:
    """Save TIFF, trying LZW compression first."""
    try:
        fig.savefig(
            out_tiff,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except TypeError:
        fig.savefig(out_tiff, dpi=dpi, bbox_inches="tight", facecolor="white")


# ---------------------------------------------------------------------
# 3. Main figure generation
# ---------------------------------------------------------------------

def make_figure(
    project_root: Path,
    mep_dir: Optional[Path],
    descriptor_path: Optional[Path],
    out_dir: Path,
    dpi: int = 600,
    trim: bool = True,
    show_homo_lumo: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptors, descriptor_source = load_descriptors(descriptor_path)

    image_paths = {}
    images = {}

    for cid in COMPOUNDS:
        p = find_mep_image(cid, project_root=project_root, mep_dir=mep_dir)
        image_paths[cid] = p

        img = Image.open(p).convert("RGBA")
        if trim:
            img = trim_whitespace(img)
        images[cid] = img

    # Manuscript-style font embedding
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.20, 7.40),
        dpi=dpi,
        constrained_layout=False,
    )

    axes_flat = axes.ravel()

    for ax, cid in zip(axes_flat, COMPOUNDS):
        img = images[cid]
        ax.imshow(img)
        ax.set_axis_off()

        panel = PANEL_LABELS[cid]
        role = ROLE_LABELS[cid]
        desc = descriptor_text(cid, descriptors, show_homo_lumo=show_homo_lumo)

        # Panel label + compound ID
        ax.text(
            0.00, 1.035,
            f"{panel}  {cid}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
        )

        # Short role label
        ax.text(
            1.00, 1.035,
            role,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
        )

        # Descriptor box below each image
        ax.text(
            0.50, -0.060,
            desc,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.70",
                linewidth=0.45,
                alpha=0.96,
            ),
        )

    fig.text(
        0.50, 0.018,
        "MEP surfaces: electron-density isosurface 0.004 a.u.; common ESP scale −0.08 to +0.08 a.u. "
        "DFT/MEP is used as qualitative electronic-structure diagnostic context.",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )

    plt.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.945,
        bottom=0.110,
        wspace=0.030,
        hspace=0.215,
    )

    out_png = out_dir / "Figure_7_DFT_MEP_main_panel.png"
    out_pdf = out_dir / "Figure_7_DFT_MEP_main_panel.pdf"
    out_tiff = out_dir / "Figure_7_DFT_MEP_main_panel.tiff"
    out_manifest = out_dir / "Figure_7_DFT_MEP_main_panel_manifest.txt"

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")
    save_tiff(fig, out_tiff, dpi=dpi)
    plt.close(fig)

    with open(out_manifest, "w", encoding="utf-8") as f:
        f.write("Figure 7 DFT/MEP main panel generation manifest\n")
        f.write("=" * 60 + "\n")
        f.write(f"Project root: {project_root}\n")
        f.write(f"MEP directory argument: {mep_dir}\n")
        f.write(f"Descriptor source: {descriptor_source}\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"DPI: {dpi}\n")
        f.write(f"Trim whitespace: {trim}\n")
        f.write(f"Show HOMO/LUMO in panel labels: {show_homo_lumo}\n\n")

        f.write("Input image files:\n")
        for cid in COMPOUNDS:
            f.write(f"  {cid}: {image_paths[cid]}\n")

        f.write("\nDescriptors used:\n")
        for cid in COMPOUNDS:
            f.write(f"  {cid}: {descriptors[cid]}\n")

        f.write("\nOutput files:\n")
        f.write(f"  {out_png}\n")
        f.write(f"  {out_pdf}\n")
        f.write(f"  {out_tiff}\n")

    print("[OK] Figure 7 files written:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")
    print(f"  {out_tiff}")
    print(f"  {out_manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 7 DFT/MEP main panel for matched N = 2 coumarins."
    )

    parser.add_argument(
        "--project-root",
        default=r"D:\Makaleler\coumarin-logp-working-source",
        help="Project root folder.",
    )

    parser.add_argument(
        "--mep-dir",
        default=None,
        help="Folder containing the four MEP PNG/TIFF files. If omitted, script searches likely folders.",
    )

    parser.add_argument(
        "--descriptor-csv",
        default=None,
        help="CSV/XLSX descriptor table. If omitted, script tries to auto-detect; otherwise fallback values are used.",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder. Default: <project-root>\\figures\\manuscript",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output resolution for PNG/TIFF.",
    )

    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable automatic whitespace trimming around MEP images.",
    )

    parser.add_argument(
        "--show-homo-lumo",
        action="store_true",
        help="Also show HOMO and LUMO values in the panel annotations. Usually too crowded for main text.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    if not project_root.exists():
        print(f"[ERROR] Project root does not exist: {project_root}")
        sys.exit(1)

    mep_dir = Path(args.mep_dir) if args.mep_dir else None
    if mep_dir is not None and not mep_dir.exists():
        print(f"[ERROR] MEP directory does not exist: {mep_dir}")
        sys.exit(1)

    if args.descriptor_csv:
        descriptor_path = Path(args.descriptor_csv)
    else:
        descriptor_path = candidate_descriptor_path(project_root)
        if descriptor_path is not None:
            print(f"[INFO] Auto-detected descriptor table: {descriptor_path}")
        else:
            print("[WARNING] No descriptor table auto-detected. Fallback values will be used.")

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "figures" / "manuscript"

    make_figure(
        project_root=project_root,
        mep_dir=mep_dir,
        descriptor_path=descriptor_path,
        out_dir=out_dir,
        dpi=args.dpi,
        trim=not args.no_trim,
        show_homo_lumo=args.show_homo_lumo,
    )


if __name__ == "__main__":
    main()