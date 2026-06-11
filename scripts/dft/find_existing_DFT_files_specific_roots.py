from pathlib import Path
import csv
import re

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")

SEARCH_ROOTS = [
    Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp"),
    Path(r"C:\orca_tests\DFT_FINAL_CLEAN"),
    Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES"),
]

TARGETS = ["CMR_GOLD_043", "CMR_GOLD_055", "CMR_GOLD_058", "CMR_GOLD_079"]

OUTDIR = PROJECT / "audit_existing"
OUTDIR.mkdir(parents=True, exist_ok=True)

INVENTORY_CSV = OUTDIR / "specific_roots_DFT_file_inventory_043_055_058_079.csv"
EXTRACTED_CSV = OUTDIR / "specific_roots_DFT_extracted_values_043_055_058_079.csv"
SUMMARY_TXT = OUTDIR / "specific_roots_DFT_file_summary_043_055_058_079.txt"

INTEREST_EXTS = {
    ".out", ".inp", ".gbw", ".xyz", ".csv", ".xlsx", ".cube",
    ".densities", ".molden", ".input", ".png", ".jpg", ".jpeg",
    ".sdf", ".smi", ".log", ".txt"
}


def classify_file(path: Path):
    name = path.name.lower()

    if name.endswith(".out"):
        if "optfreq" in name or "_opt" in name:
            return "ORCA opt/freq output"
        if "_sp" in name or "single" in name:
            return "ORCA SP output"
        return "ORCA output / unknown type"

    if name.endswith(".inp"):
        if "optfreq" in name or "_opt" in name:
            return "ORCA opt/freq input"
        if "_sp" in name:
            return "ORCA SP input"
        return "ORCA input / unknown type"

    if name.endswith(".gbw"):
        if "_sp" in name:
            return "SP GBW"
        if "optfreq" in name or "_opt" in name:
            return "Opt/Freq GBW"
        return "GBW"

    if name.endswith(".densities"):
        return "ORCA densities container"

    if name.endswith(".cube"):
        if "esp" in name or "potential" in name or "elpot" in name:
            return "ESP cube"
        if "eldens" in name or "density" in name or "dens" in name:
            return "Density cube"
        return "Cube"

    if name.endswith(".xyz"):
        if "_opt" in name:
            return "Optimized XYZ"
        if "start" in name:
            return "Starting XYZ"
        return "XYZ"

    if name.endswith(".csv"):
        if "summary" in name or "sp_summary" in name:
            return "Extracted descriptor CSV"
        if "charge" in name:
            return "Charge CSV"
        return "CSV"

    if name.endswith((".png", ".jpg", ".jpeg")):
        return "Figure/image"

    if name.endswith(".molden") or name.endswith(".molden.input"):
        return "Molden file"

    if name.endswith(".sdf"):
        return "SDF structure"

    if name.endswith(".smi"):
        return "SMILES file"

    return "Other"


def parse_orca_out(path: Path):
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    normal = "ORCA TERMINATED NORMALLY" in text
    converged = (
        "THE OPTIMIZATION HAS CONVERGED" in text
        or "HURRAY" in text
        or "OPTIMIZATION RUN DONE" in text
    )

    route_lines = [l.strip() for l in lines if l.strip().startswith("!")]

    freqs = []
    in_freq = False

    for line in lines:
        if "VIBRATIONAL FREQUENCIES" in line:
            in_freq = True
            continue

        if in_freq:
            if "NORMAL MODES" in line or "IR SPECTRUM" in line:
                break

            m = re.match(r"\s*\d+\s*:\s*(-?\d+\.\d+)\s*cm\*\*-1", line)
            if m:
                freqs.append(float(m.group(1)))

    imag = [f for f in freqs if f < -20.0]

    orbital_rows = []
    in_orb = False

    for line in lines:
        if "ORBITAL ENERGIES" in line:
            in_orb = True
            continue

        if in_orb:
            if "MOLECULAR ORBITALS" in line or "MULLIKEN" in line or "LOEWDIN" in line:
                if orbital_rows:
                    break

            parts = line.split()
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    occ = float(parts[1])
                    e_hartree = float(parts[2])
                    e_ev = float(parts[3])
                    orbital_rows.append((idx, occ, e_hartree, e_ev))
                except ValueError:
                    pass

    homo = None
    lumo = None

    for row in orbital_rows:
        idx, occ, eh, ev = row
        if occ > 0.0:
            homo = row
        elif occ == 0.0 and homo is not None:
            lumo = row
            break

    dipole = None

    for line in lines:
        if "Magnitude (Debye)" in line:
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if nums:
                dipole = float(nums[-1])

    return {
        "normal_termination": normal,
        "optimization_converged": converged,
        "frequency_count": len(freqs),
        "imaginary_frequencies_lt_minus20": ";".join(str(x) for x in imag),
        "HOMO_eV": homo[3] if homo else "",
        "LUMO_eV": lumo[3] if lumo else "",
        "Gap_eV": (lumo[3] - homo[3]) if homo and lumo else "",
        "Dipole_Debye": dipole if dipole is not None else "",
        "route_line": route_lines[0] if route_lines else "",
    }


def main():
    inventory_rows = []
    extracted_rows = []

    print("Target roots:")
    for root in SEARCH_ROOTS:
        print(f"  {root} | exists={root.exists()}")
    print()

    for root in SEARCH_ROOTS:
        if not root.exists():
            print(f"UYARI: Klasör bulunamadı, atlandı: {root}")
            continue

        print(f"Taranıyor: {root}")

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            name = path.name
            full = str(path)

            target_hit = None
            for target in TARGETS:
                if target in name or target in full:
                    target_hit = target
                    break

            if not target_hit:
                continue

            ext = path.suffix.lower()
            if ext not in INTEREST_EXTS and not name.lower().endswith(".molden.input"):
                continue

            try:
                size = path.stat().st_size
            except OSError:
                size = ""

            file_type = classify_file(path)

            inventory_rows.append({
                "Compound_ID": target_hit,
                "Root": str(root),
                "File_type": file_type,
                "File_name": name,
                "Path": str(path),
                "Size_bytes": size,
            })

            if path.suffix.lower() == ".out":
                try:
                    parsed = parse_orca_out(path)
                    extracted_rows.append({
                        "Compound_ID": target_hit,
                        "Root": str(root),
                        "File_name": name,
                        "Path": str(path),
                        "File_type": file_type,
                        **parsed,
                    })
                except Exception as e:
                    extracted_rows.append({
                        "Compound_ID": target_hit,
                        "Root": str(root),
                        "File_name": name,
                        "Path": str(path),
                        "File_type": file_type,
                        "parse_error": str(e),
                    })

    inventory_rows.sort(key=lambda r: (r["Compound_ID"], r["File_type"], r["Path"]))
    extracted_rows.sort(key=lambda r: (r["Compound_ID"], r["File_name"], r["Path"]))

    with INVENTORY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["Compound_ID", "Root", "File_type", "File_name", "Path", "Size_bytes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory_rows)

    extracted_fieldnames = [
        "Compound_ID", "Root", "File_name", "Path", "File_type",
        "normal_termination", "optimization_converged", "frequency_count",
        "imaginary_frequencies_lt_minus20", "HOMO_eV", "LUMO_eV",
        "Gap_eV", "Dipole_Debye", "route_line", "parse_error"
    ]

    with EXTRACTED_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=extracted_fieldnames)
        writer.writeheader()
        for row in extracted_rows:
            writer.writerow({k: row.get(k, "") for k in extracted_fieldnames})

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("Specific-root DFT file summary for CMR_GOLD_043, 055, 058, 079\n")
        f.write("=" * 100 + "\n\n")

        f.write("Search roots:\n")
        for root in SEARCH_ROOTS:
            f.write(f"- {root} | exists={root.exists()}\n")

        for target in TARGETS:
            target_files = [r for r in inventory_rows if r["Compound_ID"] == target]
            target_outs = [r for r in extracted_rows if r["Compound_ID"] == target]

            f.write(f"\n\n{target}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total relevant files found: {len(target_files)}\n")

            type_counts = {}
            for r in target_files:
                type_counts[r["File_type"]] = type_counts.get(r["File_type"], 0) + 1

            for t, c in sorted(type_counts.items()):
                f.write(f"  {t}: {c}\n")

            f.write("\nORCA output candidates:\n")
            if not target_outs:
                f.write("  No .out files found.\n")
            else:
                for r in target_outs:
                    f.write(f"  {r.get('File_name')}\n")
                    f.write(f"    Path: {r.get('Path')}\n")
                    f.write(f"    Type: {r.get('File_type')}\n")
                    f.write(f"    Normal: {r.get('normal_termination')}\n")
                    f.write(f"    Converged: {r.get('optimization_converged')}\n")
                    f.write(f"    Freq count: {r.get('frequency_count')}\n")
                    f.write(f"    Imag: {r.get('imaginary_frequencies_lt_minus20')}\n")
                    f.write(f"    HOMO: {r.get('HOMO_eV')}\n")
                    f.write(f"    LUMO: {r.get('LUMO_eV')}\n")
                    f.write(f"    Gap : {r.get('Gap_eV')}\n")
                    f.write(f"    Dipole: {r.get('Dipole_Debye')}\n")
                    f.write(f"    Route: {r.get('route_line')}\n")
                    f.write("\n")

    print("\nBitti.")
    print(f"Inventory CSV : {INVENTORY_CSV}")
    print(f"Extracted CSV : {EXTRACTED_CSV}")
    print(f"Summary TXT   : {SUMMARY_TXT}")
    print()

    for target in TARGETS:
        n_files = sum(1 for r in inventory_rows if r["Compound_ID"] == target)
        n_outs = sum(1 for r in extracted_rows if r["Compound_ID"] == target)
        print(f"{target}: {n_files} relevant files, {n_outs} ORCA .out candidates")

    print("\nÖzeti görmek için:")
    print(f'type "{SUMMARY_TXT}"')


if __name__ == "__main__":
    main()