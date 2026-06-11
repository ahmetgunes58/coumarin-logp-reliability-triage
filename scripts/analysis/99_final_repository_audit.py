# -*- coding: utf-8 -*-
"""
Final repository audit for coumarin-logp coumarin logP project.

Run from project root:
    python _repository_prep\\99_final_repository_audit.py

Outputs:
    _repository_prep\\final_repository_audit_report.txt
"""

from __future__ import annotations

from pathlib import Path
import re
import zipfile
import html


ROOT = Path.cwd()
REPORT_PATH = ROOT / "_repository_prep" / "final_repository_audit_report.txt"

EXPECTED_FILES = [
    "figures/supporting/Figure_S5_auxiliary_MEP_maps.png",
    "figures/supporting/Figure_S5_auxiliary_MEP_maps.tiff",
    "figures/supporting/Figure_S5_auxiliary_MEP_maps.pdf",
    "data/processed/Figure_S5_auxiliary_MEP_maps_summary.txt",
    "scripts/make_figure_S5_auxiliary_MEP_maps.py",
]

FORBIDDEN_PATTERNS = [
    "*trial*",
    "*3Dpose*",
    "*topdown_protocol*",
]

SEARCH_DIRS_FOR_FORBIDDEN = [
    "figures",
    "data/processed",
    "scripts",
]

JUNK_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "Thumbs.db",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def docx_to_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return ""

    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def warn_status(ok: bool) -> str:
    return "OK" if ok else "WARN"


def find_latest_docx(prefix: str) -> Path | None:
    candidates = sorted(ROOT.glob(f"{prefix}*.docx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    lines = []
    errors = 0
    warnings = 0

    lines.append("=" * 80)
    lines.append("FINAL REPOSITORY AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"Project root: {ROOT}")
    lines.append("")

    # ------------------------------------------------------------------
    # 1. Expected files
    # ------------------------------------------------------------------
    lines.append("1. EXPECTED FINAL FILES")
    lines.append("-" * 80)

    for rel in EXPECTED_FILES:
        path = ROOT / rel
        ok = path.exists()
        if not ok:
            errors += 1
            size = "missing"
        else:
            size = f"{path.stat().st_size:,} bytes"

        lines.append(f"[{status(ok)}] {rel} | {size}")

    lines.append("")

    # ------------------------------------------------------------------
    # 2. Forbidden trial/artifact files
    # ------------------------------------------------------------------
    lines.append("2. TRIAL / TEMPORARY FILE CHECK")
    lines.append("-" * 80)

    forbidden_hits = []

    for rel_dir in SEARCH_DIRS_FOR_FORBIDDEN:
        base = ROOT / rel_dir
        if not base.exists():
            continue

        for pattern in FORBIDDEN_PATTERNS:
            for hit in base.rglob(pattern):
                if hit.is_file():
                    forbidden_hits.append(hit.relative_to(ROOT))

    if forbidden_hits:
        errors += len(forbidden_hits)
        for hit in forbidden_hits:
            lines.append(f"[FAIL] forbidden trial/temp file remains: {hit}")
    else:
        lines.append("[PASS] No trial / 3Dpose / topdown_protocol files found in figures, data/processed, or scripts.")

    lines.append("")

    # ------------------------------------------------------------------
    # 3. Figure S5 summary validation
    # ------------------------------------------------------------------
    lines.append("3. FIGURE S5 SUMMARY VALIDATION")
    lines.append("-" * 80)

    summary_path = ROOT / "data/processed/Figure_S5_auxiliary_MEP_maps_summary.txt"

    if not summary_path.exists():
        errors += 1
        lines.append("[FAIL] Figure S5 summary is missing.")
    else:
        summary = read_text(summary_path)

        checks = [
            ("No 'trial' string in summary", "trial" not in summary.lower()),
            ("Surface isovalue = 0.004000 a.u.", "Surface isovalue: 0.004000 a.u." in summary),
            ("Common ESP scale = -0.080 to +0.080 a.u.", "Common ESP scale: -0.080 to +0.080 a.u." in summary),
            ("Perspective projection = False", "Perspective projection: False" in summary),
            ("CMR-GOLD-079 present", "CMR-GOLD-079" in summary),
            ("CMR-GOLD-079 rot_x = 20.000", "rot_x=20.000" in summary),
            ("CMR-GOLD-079 rot_y = 0.000", "rot_y=0.000" in summary),
            ("CMR-GOLD-079 rot_z = 8.000", "rot_z=8.000" in summary),
            ("CMR-GOLD-079 formula = C19H14N4O4", "formula      : C19H14N4O4" in summary),
            ("CMR-GOLD-079 components = 1", "components   : 1" in summary),
        ]

        for label, ok in checks:
            if not ok:
                errors += 1
            lines.append(f"[{status(ok)}] {label}")

    lines.append("")

    # ------------------------------------------------------------------
    # 4. Figure S5 script validation
    # ------------------------------------------------------------------
    lines.append("4. FIGURE S5 SCRIPT VALIDATION")
    lines.append("-" * 80)

    script_path = ROOT / "scripts/make_figure_S5_auxiliary_MEP_maps.py"

    if not script_path.exists():
        errors += 1
        lines.append("[FAIL] Final Figure S5 script is missing.")
    else:
        script = read_text(script_path)

        script_checks = [
            ("Final output PNG name", 'Figure_S5_auxiliary_MEP_maps.png' in script),
            ("Final output TIFF name", 'Figure_S5_auxiliary_MEP_maps.tiff' in script),
            ("Final output PDF name", 'Figure_S5_auxiliary_MEP_maps.pdf' in script),
            ("Final output summary name", 'Figure_S5_auxiliary_MEP_maps_summary.txt' in script),
            ("Surface isovalue set to 0.004", "SURFACE_ISOVALUE = 0.004" in script),
            ("Global ESP clip set to 0.080", "GLOBAL_CLIP = 0.080" in script),
            ("CMR_GOLD_079 rot_x 20.0 present", '"rot_x": 20.0' in script or '"rot_x": 20' in script),
            ("CMR_GOLD_079 rot_y 0.0 present", '"rot_y": 0.0' in script or '"rot_y": 0' in script),
            ("CMR_GOLD_079 rot_z 8.0 present", '"rot_z": 8.0' in script or '"rot_z": 8' in script),
            ("Surface opacity final value 0.48", "SURFACE_OPACITY = 0.48" in script),
            ("Surface ambient final value 0.14", "SURFACE_AMBIENT = 0.14" in script),
            ("Surface diffuse final value 0.80", "SURFACE_DIFFUSE = 0.80" in script),
            ("Surface specular final value 0.02", "SURFACE_SPECULAR = 0.02" in script),
            ("Surface specular power final value 8", "SURFACE_SPECULAR_POWER = 8" in script),
        ]

        for label, ok in script_checks:
            if not ok:
                errors += 1
            lines.append(f"[{status(ok)}] {label}")

    lines.append("")

    # ------------------------------------------------------------------
    # 5. Manuscript / SI text validation
    # ------------------------------------------------------------------
    lines.append("5. MANUSCRIPT AND SI TEXT VALIDATION")
    lines.append("-" * 80)

    manuscript = find_latest_docx("Manuscript_coumarin-logp_Coumarin_logP")
    si = find_latest_docx("Supporting_Information_coumarin-logp_Coumarin_logP")

    if manuscript is None:
        warnings += 1
        lines.append("[WARN] Manuscript docx not found in project root.")
    else:
        m_text = docx_to_text(manuscript)
        lines.append(f"[OK] Manuscript detected: {manuscript.name}")

        manuscript_checks = [
            ("Manuscript cites Figure S5", "Figure S5" in m_text),
            ("Manuscript contains approximately 85° geometry statement", "approximately 85" in m_text),
            ("Manuscript contains fragment-local electronic effects statement", "fragment-local electronic effects" in m_text),
            ("Manuscript contains CMR_GOLD_079", "CMR_GOLD_079" in m_text),
        ]

        for label, ok in manuscript_checks:
            if not ok:
                warnings += 1
            lines.append(f"[{warn_status(ok)}] {label}")

    if si is None:
        warnings += 1
        lines.append("[WARN] Supporting Information docx not found in project root.")
    else:
        si_text = docx_to_text(si)
        lines.append(f"[OK] Supporting Information detected: {si.name}")

        si_checks = [
            ("SI cites Figure S5", "Figure S5" in si_text),
            ("SI Figure S5 caption contains non-planar DFT-optimised geometry", "non-planar DFT-optimised geometry" in si_text),
            ("SI mentions MEP scale for Figures 7 and S5", "MEP scale for Figures 7 and S5" in si_text),
            ("SI contains isovalue = 0.004", "0.004" in si_text),
            ("SI contains -0.08 to +0.08", "-0.08 to +0.08" in si_text or "−0.08 to +0.08" in si_text),
        ]

        for label, ok in si_checks:
            if not ok:
                warnings += 1
            lines.append(f"[{warn_status(ok)}] {label}")

    lines.append("")

    # ------------------------------------------------------------------
    # 6. Junk files
    # ------------------------------------------------------------------
    lines.append("6. JUNK FILE CHECK")
    lines.append("-" * 80)

    junk_hits = []

    for pattern in JUNK_PATTERNS:
        for hit in ROOT.rglob(pattern):
            if "_repository_prep" in hit.parts:
                continue
            junk_hits.append(hit.relative_to(ROOT))

    if junk_hits:
        warnings += len(junk_hits)
        for hit in junk_hits:
            lines.append(f"[WARN] junk/cache file detected: {hit}")
    else:
        lines.append("[OK] No common junk/cache files detected.")

    lines.append("")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    lines.append("=" * 80)
    lines.append("AUDIT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Errors  : {errors}")
    lines.append(f"Warnings: {warnings}")

    if errors == 0:
        lines.append("FINAL STATUS: PASS FOR REPOSITORY EXPORT, subject to review of warnings.")
    else:
        lines.append("FINAL STATUS: DO NOT EXPORT YET. Resolve FAIL items first.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"Audit report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()