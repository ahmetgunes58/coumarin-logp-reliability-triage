from pathlib import Path
import re

root = Path(r"D:\Makaleler\coumarin-logp-working-source")

targets = [
    "CMR_GOLD_055", "CMR_GOLD_043", "CMR_GOLD_044", "CMR_GOLD_029", "CMR_GOLD_058",
    "CMR_GOLD_079", "CMR_GOLD_016", "CMR_GOLD_090", "CMR_GOLD_020", "CMR_GOLD_092"
]

keywords = [
    "B3LYP", "PBE0", "D3BJ", "D4", "def2-SVP", "def2-TZVP",
    "CPCM", "WATER", "RIJCOSX", "TightSCF", "FINAL SINGLE POINT ENERGY",
    "ORCA VERSION"
]

out_files = list(root.rglob("*.out")) + list(root.rglob("*.inp"))

for target in targets:
    print("\n" + "="*90)
    print(target)
    print("="*90)

    matched = [p for p in out_files if target.lower() in str(p).lower()]

    if not matched:
        print("No matching .out/.inp files found.")
        continue

    for p in matched:
        print(f"\nFILE: {p}")
        try:
            text = p.read_text(errors="ignore")
        except Exception as e:
            print(f"Could not read file: {e}")
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            upper = line.upper()
            if any(k.upper() in upper for k in keywords):
                print(f"  L{i}: {line.strip()}")