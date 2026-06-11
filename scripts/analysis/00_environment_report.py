# -*- coding: utf-8 -*-
"""
Generate software environment report for the coumarin-logp repository.

Run separately in each conda environment used in the project, e.g.:

    conda activate base
    python scripts\\00_environment_report.py

    conda activate mepfig
    python scripts\\00_environment_report.py

Outputs:
    data/processed/Dataset_S18_software_environment_<env>.csv
    data/processed/Dataset_S18_software_environment_<env>.txt
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_NAME = os.environ.get("CONDA_DEFAULT_ENV", "unknown_env")


PACKAGES = {
    "Python": None,
    "pandas": "pandas",
    "NumPy": "numpy",
    "SciPy": "scipy",
    "matplotlib": "matplotlib",
    "RDKit": "rdkit",
    "PyVista": "pyvista",
}


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed / not available"


def orca_version() -> str:
    try:
        result = subprocess.run(
            ["orca", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (result.stdout or result.stderr).strip()
        return text.splitlines()[0] if text else "not found in PATH"
    except Exception:
        return "not found in PATH"


def main() -> None:
    rows = []

    rows.append(("Conda environment", ENV_NAME))
    rows.append(("Operating system", platform.platform()))
    rows.append(("Python", sys.version.replace("\n", " ")))

    for label, package_name in PACKAGES.items():
        if package_name is None:
            continue
        rows.append((label, package_version(package_name)))

    rows.append(("ORCA", orca_version()))

    csv_path = OUT_DIR / f"Dataset_S18_software_environment_{ENV_NAME}.csv"
    txt_path = OUT_DIR / f"Dataset_S18_software_environment_{ENV_NAME}.txt"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Component", "Version / details"])
        writer.writerows(rows)

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Software environment report\n")
        handle.write("=" * 72 + "\n")
        for component, version in rows:
            handle.write(f"{component}: {version}\n")

    print("\nSoftware environment report completed.")
    print(f"Environment: {ENV_NAME}")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
    print("\nDetected versions:")
    for component, version in rows:
        print(f"{component}: {version}")


if __name__ == "__main__":
    main()