from pathlib import Path
import csv

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
AUDIT = PROJECT / "audit_existing"

EXTRACTED_CSV = AUDIT / "specific_roots_DFT_extracted_values_043_055_058_079.csv"
OUT_TXT = AUDIT / "shortlisted_DFT_outputs_043_055_058_079.txt"
OUT_CSV = AUDIT / "shortlisted_DFT_outputs_043_055_058_079.csv"

TARGETS = ["CMR_GOLD_043", "CMR_GOLD_055", "CMR_GOLD_058", "CMR_GOLD_079"]


def to_bool(x):
    return str(x).strip().lower() == "true"


def to_float_or_none(x):
    try:
        if str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def to_int_or_zero(x):
    try:
        if str(x).strip() == "":
            return 0
        return int(float(x))
    except Exception:
        return 0


def is_empty(x):
    return str(x).strip() == ""


def classify_output(row):
    name = row.get("File_name", "").lower()
    path = row.get("Path", "").lower()

    homo = to_float_or_none(row.get("HOMO_eV", ""))
    lumo = to_float_or_none(row.get("LUMO_eV", ""))
    gap = to_float_or_none(row.get("Gap_eV", ""))
    dip = to_float_or_none(row.get("Dipole_Debye", ""))

    normal = to_bool(row.get("normal_termination", ""))
    converged = to_bool(row.get("optimization_converged", ""))
    freq_count = to_int_or_zero(row.get("frequency_count", ""))
    imag = row.get("imaginary_frequencies_lt_minus20", "")

    has_orbitals = homo is not None and lumo is not None and gap is not None
    has_dipole = dip is not None

    looks_sp = (
        "_sp" in name
        or "single" in name
        or "properties" in path
        or has_orbitals
    )

    looks_opt = (
        "optfreq" in name
        or "_opt" in name
        or "freq" in name
        or converged
        or freq_count > 0
    )

    if normal and converged and freq_count > 0:
        return "OPT_FREQ"
    if normal and has_orbitals and has_dipole:
        return "SP"
    if looks_opt:
        return "OPT_FREQ_CANDIDATE"
    if looks_sp:
        return "SP_CANDIDATE"
    return "UNKNOWN"


def score_row(row, kind):
    score = 0

    name = row.get("File_name", "").lower()
    path = row.get("Path", "").lower()

    normal = to_bool(row.get("normal_termination", ""))
    converged = to_bool(row.get("optimization_converged", ""))
    freq_count = to_int_or_zero(row.get("frequency_count", ""))
    imag = row.get("imaginary_frequencies_lt_minus20", "")

    homo = to_float_or_none(row.get("HOMO_eV", ""))
    lumo = to_float_or_none(row.get("LUMO_eV", ""))
    gap = to_float_or_none(row.get("Gap_eV", ""))
    dip = to_float_or_none(row.get("Dipole_Debye", ""))

    if normal:
        score += 20

    if kind == "OPT_FREQ":
        if converged:
            score += 20
        if freq_count > 0:
            score += 15
        if is_empty(imag):
            score += 10
        if "optfreq" in name:
            score += 8
        if "_opt" in name:
            score += 5
        if "final" in path or "clean" in path:
            score += 3

    if kind == "SP":
        if homo is not None:
            score += 10
        if lumo is not None:
            score += 10
        if gap is not None:
            score += 10
        if dip is not None:
            score += 10
        if "_sp" in name:
            score += 8
        if "properties" in path:
            score += 6
        if "final" in path or "clean" in path:
            score += 3

    return score


def main():
    if not EXTRACTED_CSV.exists():
        raise FileNotFoundError(f"Önce şu dosya oluşmalı: {EXTRACTED_CSV}")

    with EXTRACTED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    output_rows = []

    for row in rows:
        cls = classify_output(row)
        row["Auto_class"] = cls

        opt_score = score_row(row, "OPT_FREQ")
        sp_score = score_row(row, "SP")

        row["Opt_score"] = opt_score
        row["SP_score"] = sp_score

        output_rows.append(row)

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(output_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("Shortlisted DFT outputs for 043 / 055 / 058 / 079\n")
        f.write("=" * 100 + "\n\n")

        for target in TARGETS:
            target_rows = [r for r in output_rows if r.get("Compound_ID") == target]

            opt_candidates = sorted(
                target_rows,
                key=lambda r: r["Opt_score"],
                reverse=True
            )[:5]

            sp_candidates = sorted(
                target_rows,
                key=lambda r: r["SP_score"],
                reverse=True
            )[:5]

            f.write(f"\n{target}\n")
            f.write("-" * 100 + "\n")

            f.write("\nTOP OPT/FREQ CANDIDATES\n")
            for i, r in enumerate(opt_candidates, 1):
                f.write(f"{i}. score={r['Opt_score']} | class={r['Auto_class']}\n")
                f.write(f"   file: {r.get('File_name')}\n")
                f.write(f"   path: {r.get('Path')}\n")
                f.write(f"   normal={r.get('normal_termination')} | converged={r.get('optimization_converged')} | freq_count={r.get('frequency_count')} | imag={r.get('imaginary_frequencies_lt_minus20')}\n")
                f.write(f"   HOMO={r.get('HOMO_eV')} | LUMO={r.get('LUMO_eV')} | Gap={r.get('Gap_eV')} | Dipole={r.get('Dipole_Debye')}\n")
                f.write("\n")

            f.write("\nTOP SP / PROPERTY CANDIDATES\n")
            for i, r in enumerate(sp_candidates, 1):
                f.write(f"{i}. score={r['SP_score']} | class={r['Auto_class']}\n")
                f.write(f"   file: {r.get('File_name')}\n")
                f.write(f"   path: {r.get('Path')}\n")
                f.write(f"   normal={r.get('normal_termination')} | converged={r.get('optimization_converged')} | freq_count={r.get('frequency_count')} | imag={r.get('imaginary_frequencies_lt_minus20')}\n")
                f.write(f"   HOMO={r.get('HOMO_eV')} | LUMO={r.get('LUMO_eV')} | Gap={r.get('Gap_eV')} | Dipole={r.get('Dipole_Debye')}\n")
                f.write("\n")

    print("Kısa liste oluşturuldu.")
    print(OUT_TXT)
    print(OUT_CSV)

    print("\nÖzet için şu komutu çalıştır:")
    print(f'type "{OUT_TXT}"')


if __name__ == "__main__":
    main()