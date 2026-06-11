# -*- coding: utf-8 -*-
"""
25a_scan_figure_S5_mep_readiness.py

Scans local project and external ORCA folders for Figure S5 MEP readiness.

Targets:
CMR_GOLD_055, CMR_GOLD_079, CMR_GOLD_016, CMR_GOLD_090, CMR_GOLD_020, CMR_GOLD_092

Run:
python scripts\\25a_scan_figure_S5_mep_readiness.py
"""

from pathlib import Path
import csv

PROJECT_ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
EXTERNAL_ROOTS = [
    Path(r"C:\orca_tests\DFT_FINAL_CLEAN"),
]

OUTDIR = PROJECT_ROOT / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "CMR_GOLD_055",
    "CMR_GOLD_079",
    "CMR_GOLD_016",
    "CMR_GOLD_090",
    "CMR_GOLD_020",
    "CMR_GOLD_092",
]

KEY_EXTS = {".xyz", ".gbw", ".out", ".cube", ".cub", ".png", ".inp", ".densities", ".densitiesinfo"}


def normal_termination(out_file: Path) -> str:
    if not out_file.exists():
        return "missing"
    try:
        txt = out_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "unreadable"
    return "yes" if "ORCA TERMINATED NORMALLY" in txt else "no_or_unknown"


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def search_external(target: str):
    hits = []
    for root in EXTERNAL_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and target.lower() in str(p).lower() and p.suffix.lower() in KEY_EXTS:
                hits.append(p)
    return sorted(hits, key=lambda x: str(x).lower())


def classify_external(hits):
    density = []
    esp = []
    gbw = []
    xyz = []
    out = []
    png = []

    for p in hits:
        name = p.name.lower()
        suffix = p.suffix.lower()

        if suffix in {".cube", ".cub"}:
            if "eldens" in name or "density" in name or "dens" in name:
                density.append(p)
            if "esp" in name:
                esp.append(p)
        elif suffix == ".gbw":
            gbw.append(p)
        elif suffix == ".xyz":
            xyz.append(p)
        elif suffix == ".out":
            out.append(p)
        elif suffix == ".png":
            png.append(p)

    return {
        "external_density": density,
        "external_esp": esp,
        "external_gbw": gbw,
        "external_xyz": xyz,
        "external_out": out,
        "external_png": png,
    }


def local_paths(target: str):
    mol_dir = PROJECT_ROOT / "dft" / "molecules" / target

    return {
        "mol_dir": mol_dir,
        "local_xyz_sp": mol_dir / "geometry" / f"{target}_sp.xyz",
        "local_xyz_opt": mol_dir / "geometry" / f"{target}_opt.xyz",
        "local_density": mol_dir / "cubes" / f"{target}_density.cube",
        "local_esp": mol_dir / "cubes" / f"{target}_esp.cube",
        "local_sp_gbw_input": mol_dir / "input" / f"{target}_sp.gbw",
        "local_sp_gbw_geometry": mol_dir / "geometry" / f"{target}_sp.gbw",
        "local_sp_out": mol_dir / "output" / f"{target}_sp.out",
        "local_opt_out": mol_dir / "output" / f"{target}_optfreq.out",
    }


def main():
    rows = []
    detailed_lines = []

    for target in TARGETS:
        lp = local_paths(target)
        external_hits = search_external(target)
        ext = classify_external(external_hits)

        local_xyz = first_existing([lp["local_xyz_sp"], lp["local_xyz_opt"]])
        local_gbw = first_existing([lp["local_sp_gbw_input"], lp["local_sp_gbw_geometry"]])

        ready_local = lp["local_density"].exists() and lp["local_esp"].exists() and local_xyz is not None
        can_copy_external = len(ext["external_density"]) > 0 and len(ext["external_esp"]) > 0
        can_generate_local = local_gbw is not None and lp["local_sp_out"].exists()

        if ready_local:
            status = "READY_LOCAL"
        elif can_copy_external:
            status = "COPY_EXTERNAL_CUBES"
        elif can_generate_local:
            status = "GENERATE_CUBES_FROM_LOCAL_GBW"
        else:
            status = "NEEDS_GBW_OR_RERUN_SP"

        row = {
            "target": target,
            "status": status,
            "local_xyz": str(local_xyz) if local_xyz else "",
            "local_density_exists": lp["local_density"].exists(),
            "local_esp_exists": lp["local_esp"].exists(),
            "local_gbw": str(local_gbw) if local_gbw else "",
            "local_sp_out": str(lp["local_sp_out"]) if lp["local_sp_out"].exists() else "",
            "local_sp_normal": normal_termination(lp["local_sp_out"]),
            "external_density_count": len(ext["external_density"]),
            "external_esp_count": len(ext["external_esp"]),
            "external_gbw_count": len(ext["external_gbw"]),
            "external_xyz_count": len(ext["external_xyz"]),
            "best_external_density": str(ext["external_density"][0]) if ext["external_density"] else "",
            "best_external_esp": str(ext["external_esp"][0]) if ext["external_esp"] else "",
            "best_external_gbw": str(ext["external_gbw"][0]) if ext["external_gbw"] else "",
        }
        rows.append(row)

        detailed_lines.append("")
        detailed_lines.append("=" * 90)
        detailed_lines.append(target)
        detailed_lines.append("-" * 90)
        detailed_lines.append(f"STATUS: {status}")
        detailed_lines.append(f"Local XYZ: {row['local_xyz']}")
        detailed_lines.append(f"Local density exists: {row['local_density_exists']}")
        detailed_lines.append(f"Local ESP exists: {row['local_esp_exists']}")
        detailed_lines.append(f"Local GBW: {row['local_gbw']}")
        detailed_lines.append(f"Local SP normal: {row['local_sp_normal']}")
        detailed_lines.append("")
        detailed_lines.append("External density cube candidates:")
        for p in ext["external_density"][:10]:
            detailed_lines.append(f"  {p}")
        detailed_lines.append("External ESP cube candidates:")
        for p in ext["external_esp"][:10]:
            detailed_lines.append(f"  {p}")
        detailed_lines.append("External GBW candidates:")
        for p in ext["external_gbw"][:10]:
            detailed_lines.append(f"  {p}")

    csv_path = OUTDIR / "FigureS5_MEP_readiness_summary.csv"
    txt_path = OUTDIR / "FigureS5_MEP_readiness_summary.txt"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    txt_path.write_text("\n".join(detailed_lines), encoding="utf-8")

    print("[OK] Figure S5 readiness scan completed.")
    print(csv_path)
    print(txt_path)
    print("")
    for row in rows:
        print(f"{row['target']}: {row['status']}")


if __name__ == "__main__":
    main()