# -*- coding: utf-8 -*-
"""
24a_scan_figure7_paths.py

Scan the project folder for Figure 7 MEP/ESP image files and DFT descriptor tables.

Run:
python scripts\\24a_scan_figure7_paths.py
"""

from pathlib import Path
import csv
import re

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_IDS = [
    "CMR_GOLD_043",
    "CMR_GOLD_044",
    "CMR_GOLD_029",
    "CMR_GOLD_058",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TABLE_EXTS = {".csv", ".xlsx", ".xls", ".txt"}

IMAGE_HINTS = [
    "mep", "esp", "electrostatic", "potential", "surface",
    "CMR_GOLD_043", "CMR_GOLD_044", "CMR_GOLD_029", "CMR_GOLD_058",
    "043", "044", "029", "058",
]

TABLE_HINTS = [
    "dft", "descriptor", "frontier", "homo", "lumo", "gap",
    "dipole", "mep", "esp", "charge", "hirshfeld", "mulliken",
    "s10", "s41", "dataset",
]


def score_file(path: Path, hints: list[str]) -> int:
    text = str(path).lower()
    score = 0

    for target in TARGET_IDS:
        if target.lower() in text:
            score += 10

    for hint in hints:
        if hint.lower() in text:
            score += 2

    # Prefer final/manuscript-like outputs
    for bonus in ["final", "manuscript", "figure", "fig", "panel", "common_scale"]:
        if bonus in text:
            score += 1

    return score


def safe_contains_targets(path: Path) -> int:
    """
    Lightweight content scan for CSV/TXT files only.
    Counts how many target IDs appear in file content.
    """
    if path.suffix.lower() not in {".csv", ".txt"}:
        return 0

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0

    return sum(1 for tid in TARGET_IDS if tid in text)


def collect_candidates():
    image_rows = []
    table_rows = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTS:
            score = score_file(path, IMAGE_HINTS)
            if score > 0:
                image_rows.append({
                    "score": score,
                    "file": str(path),
                    "name": path.name,
                    "folder": str(path.parent),
                    "extension": suffix,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                    "modified": path.stat().st_mtime,
                })

        elif suffix in TABLE_EXTS:
            score = score_file(path, TABLE_HINTS)
            content_hits = safe_contains_targets(path)
            score += content_hits * 8

            if score > 0:
                table_rows.append({
                    "score": score,
                    "target_id_hits_in_content": content_hits,
                    "file": str(path),
                    "name": path.name,
                    "folder": str(path.parent),
                    "extension": suffix,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                    "modified": path.stat().st_mtime,
                })

    image_rows.sort(key=lambda r: (-r["score"], r["file"].lower()))
    table_rows.sort(key=lambda r: (-r["score"], -r.get("target_id_hits_in_content", 0), r["file"].lower()))

    return image_rows, table_rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(image_rows, table_rows):
    summary_path = OUTDIR / "Figure7_path_scan_summary.txt"

    lines = []
    lines.append("Figure 7 path scan summary")
    lines.append("=" * 70)
    lines.append(f"Root: {ROOT}")
    lines.append("")
    lines.append("Top image candidates:")
    lines.append("-" * 70)

    for row in image_rows[:40]:
        lines.append(f"[score {row['score']:>2}] {row['file']}")

    lines.append("")
    lines.append("Top descriptor/table candidates:")
    lines.append("-" * 70)

    for row in table_rows[:40]:
        lines.append(
            f"[score {row['score']:>2}; content hits {row.get('target_id_hits_in_content', 0)}] {row['file']}"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main():
    if not ROOT.exists():
        raise SystemExit(f"Root folder does not exist: {ROOT}")

    image_rows, table_rows = collect_candidates()

    image_csv = OUTDIR / "Figure7_MEP_image_candidates.csv"
    table_csv = OUTDIR / "Figure7_descriptor_table_candidates.csv"

    write_csv(image_csv, image_rows)
    write_csv(table_csv, table_rows)
    summary_path = write_summary(image_rows, table_rows)

    print("[OK] Scan completed.")
    print(f"Image candidates: {len(image_rows)}")
    print(f"Descriptor/table candidates: {len(table_rows)}")
    print("")
    print(f"Written: {image_csv}")
    print(f"Written: {table_csv}")
    print(f"Written: {summary_path}")
    print("")
    print("Top image candidates:")
    for row in image_rows[:12]:
        print(f"  [score {row['score']}] {row['file']}")
    print("")
    print("Top descriptor/table candidates:")
    for row in table_rows[:12]:
        print(f"  [score {row['score']}; hits {row.get('target_id_hits_in_content', 0)}] {row['file']}")


if __name__ == "__main__":
    main()