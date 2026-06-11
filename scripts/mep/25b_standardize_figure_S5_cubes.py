# -*- coding: utf-8 -*-
"""
25b_standardize_figure_S5_cubes.py

Standardizes Figure S5 MEP inputs.

Targets:
- CMR_GOLD_055: copy external density/ESP cubes
- CMR_GOLD_079: copy external density/ESP cubes
- CMR_GOLD_016: generate density/ESP cubes from local GBW
- CMR_GOLD_090: generate density/ESP cubes from local GBW
- CMR_GOLD_020: generate density/ESP cubes from local GBW
- CMR_GOLD_092: generate density/ESP cubes from local GBW

Final standardized files:
dft/molecules/<ID>/geometry/<ID>_sp.xyz
dft/molecules/<ID>/cubes/<ID>_density.cube
dft/molecules/<ID>/cubes/<ID>_esp.cube

Run:
python scripts\\25b_standardize_figure_S5_cubes.py
"""

from pathlib import Path
import shutil
import subprocess
import time
import csv

PROJECT_ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
ORCA_PLOT = Path(r"C:\orca\orca_plot.exe")
READINESS_CSV = PROJECT_ROOT / "outputs" / "FigureS5_MEP_readiness_summary.csv"

EXTERNAL_ROOTS = [
    Path(r"C:\orca_tests\DFT_FINAL_CLEAN"),
]

TARGETS = [
    "CMR_GOLD_055",
    "CMR_GOLD_079",
    "CMR_GOLD_016",
    "CMR_GOLD_090",
    "CMR_GOLD_020",
    "CMR_GOLD_092",
]

KEY_EXTS = {".xyz", ".gbw", ".out", ".cube", ".cub", ".png", ".inp"}


def read_readiness_csv():
    if not READINESS_CSV.exists():
        raise FileNotFoundError(f"Missing readiness CSV: {READINESS_CSV}")

    rows = {}
    with READINESS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["target"]] = row
    return rows


def mol_paths(target):
    mol_dir = PROJECT_ROOT / "dft" / "molecules" / target
    return {
        "mol_dir": mol_dir,
        "geometry_dir": mol_dir / "geometry",
        "input_dir": mol_dir / "input",
        "output_dir": mol_dir / "output",
        "cube_dir": mol_dir / "cubes",
        "extract_dir": mol_dir / "extracted_data",
        "final_xyz": mol_dir / "geometry" / f"{target}_sp.xyz",
        "opt_xyz": mol_dir / "geometry" / f"{target}_opt.xyz",
        "final_density": mol_dir / "cubes" / f"{target}_density.cube",
        "final_esp": mol_dir / "cubes" / f"{target}_esp.cube",
        "local_sp_out": mol_dir / "output" / f"{target}_sp.out",
    }


def search_external_files(target):
    hits = []

    for root in EXTERNAL_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            s = str(p).lower()
            if target.lower() not in s:
                continue

            if p.suffix.lower() not in KEY_EXTS:
                continue

            hits.append(p)

    return sorted(hits, key=lambda x: str(x).lower())


def classify_external(hits):
    density = []
    esp = []
    xyz = []
    gbw = []

    for p in hits:
        name = p.name.lower()
        suffix = p.suffix.lower()

        if suffix in {".cube", ".cub"}:
            if "eldens" in name or "density" in name or "dens" in name:
                density.append(p)
            if "esp" in name:
                esp.append(p)

        elif suffix == ".xyz":
            xyz.append(p)

        elif suffix == ".gbw":
            gbw.append(p)

    return {
        "density": density,
        "esp": esp,
        "xyz": xyz,
        "gbw": gbw,
    }


def choose_best_external(candidates, target):
    """
    Prefer files from final/inputs_used/03_mep_cube-like folders when available.
    Otherwise use the first sorted candidate.
    """
    if not candidates:
        return None

    scored = []

    for p in candidates:
        s = str(p).lower()
        score = 0

        if "figure" in s and "input" in s:
            score += 50
        if "03_mep_cube" in s:
            score += 40
        if "v2_final" in s:
            score += 20
        if "sp_charge_clean" in s:
            score += 10
        if target.lower() in p.name.lower():
            score += 5

        scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], str(x[1]).lower()))
    return scored[0][1]


def ensure_dirs(paths):
    for key in ["geometry_dir", "cube_dir", "extract_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def standardize_xyz_from_local_or_external(target, paths, external_xyz=None):
    if external_xyz is not None and external_xyz.exists():
        shutil.copy2(external_xyz, paths["final_xyz"])
        return external_xyz

    if paths["final_xyz"].exists():
        return paths["final_xyz"]

    if paths["opt_xyz"].exists():
        shutil.copy2(paths["opt_xyz"], paths["final_xyz"])
        return paths["opt_xyz"]

    # Fallback search in molecule folder
    xyz_candidates = sorted(paths["mol_dir"].rglob("*.xyz"), key=lambda p: str(p).lower())
    if xyz_candidates:
        shutil.copy2(xyz_candidates[0], paths["final_xyz"])
        return xyz_candidates[0]

    raise FileNotFoundError(f"No XYZ file found for {target}")


def copy_external_cubes(target):
    paths = mol_paths(target)
    ensure_dirs(paths)

    hits = search_external_files(target)
    ext = classify_external(hits)

    density_src = choose_best_external(ext["density"], target)
    esp_src = choose_best_external(ext["esp"], target)
    xyz_src = choose_best_external(ext["xyz"], target)

    if density_src is None or esp_src is None:
        raise FileNotFoundError(f"External density/ESP cube not found for {target}")

    shutil.copy2(density_src, paths["final_density"])
    shutil.copy2(esp_src, paths["final_esp"])
    xyz_used = standardize_xyz_from_local_or_external(target, paths, xyz_src)

    return {
        "target": target,
        "mode": "COPY_EXTERNAL_CUBES",
        "density_src": str(density_src),
        "esp_src": str(esp_src),
        "xyz_src": str(xyz_used),
        "final_density": str(paths["final_density"]),
        "final_esp": str(paths["final_esp"]),
        "final_xyz": str(paths["final_xyz"]),
    }


def find_local_gbw(target, readiness_row):
    row_gbw = readiness_row.get("local_gbw", "").strip()
    if row_gbw:
        p = Path(row_gbw)
        if p.exists():
            return p

    paths = mol_paths(target)
    candidates = [
        paths["input_dir"] / f"{target}_sp.gbw",
        paths["geometry_dir"] / f"{target}_sp.gbw",
        paths["input_dir"] / f"{target}_optfreq.gbw",
        paths["geometry_dir"] / f"{target}_optfreq.gbw",
    ]

    for p in candidates:
        if p.exists():
            return p

    gbw_candidates = sorted(paths["mol_dir"].rglob("*.gbw"), key=lambda p: str(p).lower())
    if gbw_candidates:
        return gbw_candidates[0]

    raise FileNotFoundError(f"No local GBW found for {target}")


def check_sp_out_if_available(target):
    paths = mol_paths(target)
    out = paths["local_sp_out"]

    if not out.exists():
        return "missing"

    text = out.read_text(encoding="utf-8", errors="ignore")
    return "normal" if "ORCA TERMINATED NORMALLY" in text else "not_normal_or_unknown"


def run_orcaplot(gbw, stdin_text, workdir, log_path):
    cmd = [str(ORCA_PLOT), str(gbw), "-i"]

    proc = subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="ignore",
        timeout=1200,
    )

    log_path.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    return proc.returncode, proc.stdout


def list_cubes(workdir):
    return sorted(workdir.glob("*.cube"), key=lambda p: p.stat().st_mtime)


def copy_best_density_cube(workdir, final_density):
    candidates = []

    for p in list_cubes(workdir):
        name = p.name.lower()
        if "eldens" in name or "density" in name or "dens" in name:
            candidates.append(p)

    if not candidates:
        return None

    best = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy2(best, final_density)
    return best


def copy_best_esp_cube(workdir, final_esp):
    candidates = []

    for p in list_cubes(workdir):
        name = p.name.lower()
        if "esp" in name:
            candidates.append(p)

    if not candidates:
        return None

    best = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy2(best, final_esp)
    return best


def generate_density_cube(target, gbw, workdir, log_path, final_density):
    stdin_text = "\n".join([
        "1",
        "2",
        "y",
        "5",
        "7",
        "11",
        "12",
        "",
    ])

    rc, _ = run_orcaplot(gbw, stdin_text, workdir, log_path)
    src = copy_best_density_cube(workdir, final_density)
    return rc, src


def generate_esp_cube(target, gbw, workdir, extract_dir, final_esp):
    """
    ESP generation requires answering the density-name prompt correctly.
    Try full path first, then short scfp name, then index 0.
    """
    stem = gbw.stem
    full_density_name = str(workdir / f"{stem}.scfp")
    short_density_name = f"{stem}.scfp"

    attempts = [
        ("full_density_name", full_density_name),
        ("short_density_name", short_density_name),
        ("index_0", "0"),
    ]

    attempt_records = []

    for label, density_answer in attempts:
        stdin_text = "\n".join([
            "1",
            "43",
            density_answer,
            "5",
            "7",
            "11",
            "12",
            "",
        ])

        log_path = extract_dir / f"{target}_orcaplot_esp_{label}.log"
        rc, _ = run_orcaplot(gbw, stdin_text, workdir, log_path)
        src = copy_best_esp_cube(workdir, final_esp)

        attempt_records.append((label, rc, log_path, src))

        if src is not None:
            return src, attempt_records

    return None, attempt_records


def generate_cubes_from_local_gbw(target, readiness_row):
    paths = mol_paths(target)
    ensure_dirs(paths)

    gbw = find_local_gbw(target, readiness_row)
    workdir = gbw.parent

    xyz_used = standardize_xyz_from_local_or_external(target, paths, None)

    density_log = paths["extract_dir"] / f"{target}_orcaplot_density.log"
    density_rc, density_src = generate_density_cube(
        target=target,
        gbw=gbw,
        workdir=workdir,
        log_path=density_log,
        final_density=paths["final_density"],
    )

    esp_src, esp_attempts = generate_esp_cube(
        target=target,
        gbw=gbw,
        workdir=workdir,
        extract_dir=paths["extract_dir"],
        final_esp=paths["final_esp"],
    )

    return {
        "target": target,
        "mode": "GENERATE_CUBES_FROM_LOCAL_GBW",
        "gbw": str(gbw),
        "workdir": str(workdir),
        "sp_out_status": check_sp_out_if_available(target),
        "density_rc": density_rc,
        "density_src": str(density_src) if density_src else "",
        "esp_src": str(esp_src) if esp_src else "",
        "esp_attempts": "; ".join(
            f"{label}: rc={rc}, log={log_path.name}, src={src.name if src else 'NONE'}"
            for label, rc, log_path, src in esp_attempts
        ),
        "xyz_src": str(xyz_used),
        "final_density": str(paths["final_density"]),
        "final_esp": str(paths["final_esp"]),
        "final_xyz": str(paths["final_xyz"]),
    }


def validate_final_files():
    missing = []

    for target in TARGETS:
        paths = mol_paths(target)
        for p in [paths["final_xyz"], paths["final_density"], paths["final_esp"]]:
            if not p.exists():
                missing.append(str(p))

    return missing


def main():
    if not ORCA_PLOT.exists():
        raise FileNotFoundError(f"orca_plot not found: {ORCA_PLOT}")

    readiness = read_readiness_csv()
    records = []

    print("=" * 88)
    print("Standardizing Figure S5 cube inputs")
    print("=" * 88)

    for target in TARGETS:
        row = readiness.get(target)

        if row is None:
            raise KeyError(f"{target} not found in readiness CSV")

        status = row["status"]
        print(f"\n{target}: {status}")

        if status == "READY_LOCAL":
            paths = mol_paths(target)
            ensure_dirs(paths)
            xyz_used = standardize_xyz_from_local_or_external(target, paths, None)
            record = {
                "target": target,
                "mode": "READY_LOCAL",
                "xyz_src": str(xyz_used),
                "final_density": str(paths["final_density"]),
                "final_esp": str(paths["final_esp"]),
                "final_xyz": str(paths["final_xyz"]),
            }

        elif status == "COPY_EXTERNAL_CUBES":
            record = copy_external_cubes(target)

        elif status == "GENERATE_CUBES_FROM_LOCAL_GBW":
            record = generate_cubes_from_local_gbw(target, row)

        else:
            raise RuntimeError(f"{target} requires unavailable data or SP rerun: {status}")

        records.append(record)

        print("  mode:", record.get("mode"))
        print("  final density:", record.get("final_density"))
        print("  final ESP:", record.get("final_esp"))
        print("  final XYZ:", record.get("final_xyz"))

    missing = validate_final_files()

    report_path = PROJECT_ROOT / "outputs" / "FigureS5_cube_standardization_report.txt"
    csv_path = PROJECT_ROOT / "outputs" / "FigureS5_cube_standardization_report.csv"

    all_keys = sorted({k for r in records for k in r.keys()})

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)

    lines = []
    lines.append("Figure S5 cube standardization report")
    lines.append("=" * 88)
    lines.append("")
    for r in records:
        lines.append("-" * 88)
        lines.append(r["target"])
        for k in all_keys:
            if k in r:
                lines.append(f"{k}: {r[k]}")
        lines.append("")

    if missing:
        lines.append("MISSING FINAL FILES:")
        for p in missing:
            lines.append(f"  {p}")
    else:
        lines.append("All final Figure S5 files are present.")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 88)
    if missing:
        print("[WARNING] Missing final files:")
        for p in missing:
            print(" ", p)
    else:
        print("[OK] All Figure S5 final XYZ/density/ESP files are present.")
    print(report_path)
    print(csv_path)
    print("=" * 88)


if __name__ == "__main__":
    main()