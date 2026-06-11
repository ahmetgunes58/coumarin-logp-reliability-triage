# -*- coding: utf-8 -*-
"""
24f_generate_044_cubes_orcaplot.py

Generate density and ESP cube files for CMR_GOLD_044 from existing ORCA SP GBW
using orca_plot interactive mode.

Run from Anaconda Prompt:
python scripts\\24f_generate_044_cubes_orcaplot.py
"""

from pathlib import Path
import subprocess
import shutil
import time

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL = "CMR_GOLD_044"

ORCA_PLOT = Path(r"C:\orca\orca_plot.exe")

INPUT_DIR = ROOT / "dft" / "molecules" / MOL / "input"
OUTPUT_DIR = ROOT / "dft" / "molecules" / MOL / "output"
GEOM_DIR = ROOT / "dft" / "molecules" / MOL / "geometry"
CUBE_DIR = ROOT / "dft" / "molecules" / MOL / "cubes"
EXTRACT_DIR = ROOT / "dft" / "molecules" / MOL / "extracted_data"

GBW = INPUT_DIR / f"{MOL}_sp.gbw"
SP_OUT = OUTPUT_DIR / f"{MOL}_sp.out"

DENSITY_FINAL = CUBE_DIR / f"{MOL}_density.cube"
ESP_FINAL = CUBE_DIR / f"{MOL}_esp.cube"

DENSITY_LOG = EXTRACT_DIR / f"{MOL}_orcaplot_density_retry.log"
ESP_LOG = EXTRACT_DIR / f"{MOL}_orcaplot_esp_retry.log"
REPORT = EXTRACT_DIR / f"{MOL}_cube_generation_report.txt"


def check_prerequisites():
    missing = []

    for p in [ORCA_PLOT, GBW, SP_OUT]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise FileNotFoundError("Missing required files:\n  - " + "\n  - ".join(missing))

    text = SP_OUT.read_text(encoding="utf-8", errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in text:
        raise RuntimeError(f"SP output does not show normal ORCA termination: {SP_OUT}")

    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


def run_orcaplot(stdin_text, log_path):
    cmd = [str(ORCA_PLOT), str(GBW), "-i"]

    proc = subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        cwd=str(INPUT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="ignore",
        timeout=900,
    )

    log_path.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    return proc.returncode, proc.stdout


def list_cubes():
    return sorted(INPUT_DIR.glob("*.cube"), key=lambda p: p.stat().st_mtime)


def newer_cubes(t0):
    return [p for p in list_cubes() if p.stat().st_mtime >= t0]


def copy_best_density_cube():
    candidates = []

    for p in list_cubes():
        name = p.name.lower()
        if "eldens" in name or "dens" in name or "density" in name:
            candidates.append(p)

    if not candidates:
        return None

    best = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy2(best, DENSITY_FINAL)
    return best


def copy_best_esp_cube():
    candidates = []

    for p in list_cubes():
        name = p.name.lower()
        if "esp" in name:
            candidates.append(p)

    if not candidates:
        return None

    best = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy2(best, ESP_FINAL)
    return best


def generate_density():
    """
    orca_plot sequence:
    1  -> Enter type of plot
    2  -> SCF electron density
    y  -> accept default density name
    5  -> Select output format
    7  -> Gaussian cube
    11 -> Generate plot
    12 -> exit
    """
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

    t0 = time.time()
    rc, out = run_orcaplot(stdin_text, DENSITY_LOG)
    src = copy_best_density_cube()

    return rc, src, out


def generate_esp():
    """
    ESP generation is sensitive to the density-name prompt.
    The previous CMR_GOLD_029 log failed because this prompt was not answered.
    We therefore provide the expected SCF density name explicitly.
    """
    density_name_full = str(INPUT_DIR / f"{MOL}_sp.scfp")
    density_name_short = f"{MOL}_sp.scfp"

    attempts = [
        ("full_density_name", density_name_full),
        ("short_density_name", density_name_short),
        ("index_0", "0"),
    ]

    all_logs = []
    best_src = None

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

        log_path = EXTRACT_DIR / f"{MOL}_orcaplot_esp_retry_{label}.log"
        rc, out = run_orcaplot(stdin_text, log_path)
        all_logs.append((label, rc, log_path))

        src = copy_best_esp_cube()
        if src is not None:
            best_src = src
            shutil.copy2(log_path, ESP_LOG)
            break

    return best_src, all_logs


def copy_xyz_if_needed():
    src_candidates = [
        INPUT_DIR / f"{MOL}_sp.xyz",
        INPUT_DIR / f"{MOL}_optfreq.xyz",
        INPUT_DIR / f"{MOL}_start.xyz",
        GEOM_DIR / f"{MOL}_opt.xyz",
    ]

    dst = GEOM_DIR / f"{MOL}_sp.xyz"

    for src in src_candidates:
        if src.exists():
            GEOM_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return src, dst

    return None, dst


def main():
    check_prerequisites()

    report = []
    report.append(f"{MOL} ORCA plot cube-generation report")
    report.append("=" * 72)
    report.append(f"ORCA plot: {ORCA_PLOT}")
    report.append(f"Input dir: {INPUT_DIR}")
    report.append(f"GBW: {GBW}")
    report.append(f"SP output: {SP_OUT}")
    report.append("")

    xyz_src, xyz_dst = copy_xyz_if_needed()
    report.append(f"XYZ source copied: {xyz_src}")
    report.append(f"XYZ destination: {xyz_dst}")
    report.append("")

    print("Generating density cube...")
    density_rc, density_src, density_out = generate_density()
    report.append(f"Density return code: {density_rc}")
    report.append(f"Density source cube: {density_src}")
    report.append(f"Density final cube: {DENSITY_FINAL if DENSITY_FINAL.exists() else 'NOT CREATED'}")
    report.append(f"Density log: {DENSITY_LOG}")
    report.append("")

    print("Generating ESP cube...")
    esp_src, esp_attempts = generate_esp()
    report.append(f"ESP source cube: {esp_src}")
    report.append(f"ESP final cube: {ESP_FINAL if ESP_FINAL.exists() else 'NOT CREATED'}")
    report.append("ESP attempts:")
    for label, rc, log_path in esp_attempts:
        report.append(f"  {label}: return code {rc}; log {log_path}")
    report.append("")

    report.append("Existing cube files in input directory:")
    for p in list_cubes():
        report.append(f"  {p}")

    REPORT.write_text("\n".join(report), encoding="utf-8")

    print("")
    print("[DONE]")
    print(REPORT)
    print("")
    print("Density final exists:", DENSITY_FINAL.exists(), DENSITY_FINAL)
    print("ESP final exists:", ESP_FINAL.exists(), ESP_FINAL)


if __name__ == "__main__":
    main()