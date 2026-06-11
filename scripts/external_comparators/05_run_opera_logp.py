# -*- coding: utf-8 -*-
"""
05_run_opera_logp.py

Purpose
-------
Run OPERA 2.9 command-line logP prediction for the external
platform-independent logP comparator audit.

Input
-----
data/external_predictor_input.smi

Output
------
data/processed/OPERA_logP_raw_output.csv
data/processed/Dataset_S42d_OPERA_run_report.txt

Important
---------
This script only runs OPERA and records the raw output.
A separate processing script will standardise OPERA logP values and calculate
prediction errors.

OPERA command-line usage is based on OPERA 2.9 help:
    OPERA <input> <output> [Options]
    OPERA -s input.smi -o output.csv -e logP -v 1
"""

from pathlib import Path
import subprocess
import platform
import datetime


PROJECT_DIR = Path(__file__).resolve().parents[1]

OPERA_EXE = PROJECT_DIR / "tools" / "OPERA" / "application" / "OPERA.exe"

INPUT_SMI = PROJECT_DIR / "data" / "external_predictor_input.smi"

OUT_RAW = PROJECT_DIR / "data" / "processed" / "OPERA_logP_raw_output.csv"
OUT_REPORT = PROJECT_DIR / "data" / "processed" / "Dataset_S42d_OPERA_run_report.txt"
OUT_STDOUT = PROJECT_DIR / "data" / "processed" / "OPERA_logP_stdout.txt"
OUT_STDERR = PROJECT_DIR / "data" / "processed" / "OPERA_logP_stderr.txt"


def run_command(command):
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        shell=False,
    )
    return completed


def main() -> None:
    if not OPERA_EXE.exists():
        raise FileNotFoundError(f"OPERA.exe not found:\n{OPERA_EXE}")

    if not INPUT_SMI.exists():
        raise FileNotFoundError(f"OPERA input file not found:\n{INPUT_SMI}")

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(OPERA_EXE),
        "-s",
        str(INPUT_SMI),
        "-o",
        str(OUT_RAW),
        "-e",
        "logP",
        "-v",
        "1",
        "-c",
    ]

    completed = run_command(command)

    OUT_STDOUT.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    OUT_STDERR.write_text(completed.stderr or "", encoding="utf-8", errors="replace")

    report = []
    report.append("Dataset_S42d OPERA 2.9 logP run report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Run time: {datetime.datetime.now().isoformat(timespec='seconds')}")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Python version: {platform.python_version()}")
    report.append(f"Operating system: {platform.platform()}")
    report.append("")
    report.append("1. OPERA command")
    report.append("-" * 70)
    report.append(" ".join(f'"{x}"' if " " in x else x for x in command))
    report.append("")
    report.append("2. Input/output files")
    report.append("-" * 70)
    report.append(f"OPERA executable: {OPERA_EXE}")
    report.append(f"Input SMI file: {INPUT_SMI}")
    report.append(f"Raw OPERA output: {OUT_RAW}")
    report.append(f"Stdout file: {OUT_STDOUT}")
    report.append(f"Stderr file: {OUT_STDERR}")
    report.append("")
    report.append("3. Run status")
    report.append("-" * 70)
    report.append(f"Return code: {completed.returncode}")
    report.append(f"Raw output exists: {OUT_RAW.exists()}")
    if OUT_RAW.exists():
        report.append(f"Raw output size bytes: {OUT_RAW.stat().st_size}")
    else:
        report.append("Raw output size bytes: NA")
    report.append("")
    report.append("4. Interpretation note")
    report.append("-" * 70)
    report.append(
        "This script runs OPERA 2.9 locally for the logP endpoint only. "
        "The resulting OPERA output will be processed in the next step. "
        "OPERA is used as an external platform-independent comparator and is "
        "not used to train, recalibrate, or replace any SwissADME-associated predictor."
    )

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")

    if completed.returncode != 0:
        print("WARNING: OPERA returned a non-zero exit code.")
        print("Please inspect:")
        print(OUT_STDOUT)
        print(OUT_STDERR)
    elif not OUT_RAW.exists() or OUT_RAW.stat().st_size == 0:
        print("WARNING: OPERA finished, but raw output file is missing or empty.")
        print("Please inspect stdout/stderr files.")
    else:
        print("SUCCESS: OPERA logP run completed.")


if __name__ == "__main__":
    main()