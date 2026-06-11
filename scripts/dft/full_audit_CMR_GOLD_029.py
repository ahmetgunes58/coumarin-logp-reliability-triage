from pathlib import Path
import re
import csv
import math

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_029"

MOL_DIR = PROJECT / "dft" / "molecules" / MOL_ID

OPT_INP = MOL_DIR / "input" / f"{MOL_ID}_optfreq.inp"
SP_INP = MOL_DIR / "input" / f"{MOL_ID}_sp.inp"

OPT_OUT = MOL_DIR / "output" / f"{MOL_ID}_optfreq.out"
SP_OUT = MOL_DIR / "output" / f"{MOL_ID}_sp.out"

START_XYZ = MOL_DIR / "input" / f"{MOL_ID}_start.xyz"
OPT_XYZ = MOL_DIR / "geometry" / f"{MOL_ID}_opt.xyz"

SP_GBW_INPUT = MOL_DIR / "input" / f"{MOL_ID}_sp.gbw"
SP_DENSITIES_INPUT = MOL_DIR / "input" / f"{MOL_ID}_sp.densities"

DENSITY_CUBE = MOL_DIR / "cubes" / f"{MOL_ID}_density.cube"
ESP_CUBE = MOL_DIR / "cubes" / f"{MOL_ID}_esp.cube"
MEP_PNG = MOL_DIR / "figures" / f"{MOL_ID}_MEP_common_scale.png"

CHARGE_CSV = MOL_DIR / "extracted_data" / f"{MOL_ID}_charges_mulliken_loewdin.csv"
SP_SUMMARY_CSV = MOL_DIR / "extracted_data" / f"{MOL_ID}_sp_summary.csv"

TABLE9 = PROJECT / "tables" / "Table9_DFT_panel_verified.csv"

OUT_DIR = MOL_DIR / "audit_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT = OUT_DIR / f"{MOL_ID}_FULL_AUDIT_REPORT.txt"
REPORT_CSV = OUT_DIR / f"{MOL_ID}_FULL_AUDIT_REPORT.csv"


EXPECTED_KEYWORDS_OPT = [
    "B3LYP", "D3BJ", "def2-SVP", "TightSCF", "RIJCOSX", "def2/J", "CPCM(Water)", "Opt", "Freq"
]

EXPECTED_KEYWORDS_SP = [
    "B3LYP", "D3BJ", "def2-SVP", "TightSCF", "RIJCOSX", "def2/J", "CPCM(Water)"
]


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def parse_charge_mult(inp_text):
    for line in inp_text.splitlines():
        s = line.strip()
        if s.startswith("* xyzfile"):
            parts = s.split()
            if len(parts) >= 4:
                return parts[2], parts[3], s
    return None, None, ""


def route_line(inp_text):
    for line in inp_text.splitlines():
        s = line.strip()
        if s.startswith("!"):
            return s
    return ""


def keyword_check(text, keywords):
    return {k: (k in text) for k in keywords}


def parse_orca_out(path):
    text = read_text(path)
    lines = text.splitlines()

    normal = "ORCA TERMINATED NORMALLY" in text

    opt_converged = (
        "THE OPTIMIZATION HAS CONVERGED" in text
        or "OPTIMIZATION RUN DONE" in text
        or "HURRAY" in text
    )

    # SCF convergence evidence
    scf_converged_count = text.count("SCF CONVERGED")
    final_energy_lines = [l.strip() for l in lines if "FINAL SINGLE POINT ENERGY" in l]

    # Frequencies
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

    imag_any_negative = [f for f in freqs if f < 0.0]
    imag_lt_minus20 = [f for f in freqs if f < -20.0]

    # Orbital energies
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
                    eh = float(parts[2])
                    ev = float(parts[3])
                    orbital_rows.append((idx, occ, eh, ev))
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

    homo_ev = homo[3] if homo else None
    lumo_ev = lumo[3] if lumo else None
    gap_ev = lumo_ev - homo_ev if homo and lumo else None

    # Dipole components and magnitude
    dipole_mag = None
    dipole_components = None

    for i, line in enumerate(lines):
        if "Total Dipole Moment" in line:
            # Try next lines too
            window = "\n".join(lines[i:i+12])
            nums = re.findall(r"[-+]?\d+\.\d+", window)
            # not always safe, magnitude line extracted separately below
            pass

        if "Magnitude (Debye)" in line:
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if nums:
                dipole_mag = float(nums[-1])

    # Try to parse X/Y/Z components in Debye if present
    for i, line in enumerate(lines):
        if "Dipole Moment" in line or "DIPOLE MOMENT" in line:
            block = "\n".join(lines[i:i+30])
            # ORCA often prints X/Y/Z lines before magnitude; keep raw block for report
            break
    else:
        block = ""

    # Warnings/errors
    concerning_terms = []
    for term in ["ERROR", "ABORT", "FAILED", "not converged", "NOT CONVERGED"]:
        if term in text:
            concerning_terms.append(term)

    return {
        "normal": normal,
        "opt_converged": opt_converged,
        "scf_converged_count": scf_converged_count,
        "final_energy_count": len(final_energy_lines),
        "last_final_energy_line": final_energy_lines[-1] if final_energy_lines else "",
        "frequency_count": len(freqs),
        "min_frequency": min(freqs) if freqs else None,
        "negative_frequency_count_any": len(imag_any_negative),
        "imaginary_lt_minus20_count": len(imag_lt_minus20),
        "imaginary_lt_minus20_values": imag_lt_minus20,
        "HOMO_eV": homo_ev,
        "LUMO_eV": lumo_ev,
        "Gap_eV": gap_ev,
        "Dipole_D": dipole_mag,
        "orbital_row_count": len(orbital_rows),
        "concerning_terms": ";".join(concerning_terms),
        "dipole_block_excerpt": block[:1200],
    }


def count_xyz_atoms(path):
    if not path.exists():
        return None
    lines = path.read_text(errors="ignore").splitlines()
    if not lines:
        return None
    try:
        header_n = int(lines[0].strip())
    except Exception:
        header_n = None
    coord_lines = [l for l in lines[2:] if len(l.split()) >= 4]
    return header_n, len(coord_lines)


def parse_charges_csv(path):
    if not path.exists():
        return None

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    mull_sum = 0.0
    loew_sum = 0.0
    n_rows = []
    o_rows = []

    for r in rows:
        el = r.get("Element", "")
        try:
            m = float(r.get("Mulliken_charge", ""))
        except Exception:
            m = 0.0
        try:
            l = float(r.get("Loewdin_charge", ""))
        except Exception:
            l = 0.0

        mull_sum += m
        loew_sum += l

        if el == "N":
            n_rows.append((r.get("Atom_index_1based"), el, m, l))
        if el == "O":
            o_rows.append((r.get("Atom_index_1based"), el, m, l))

    return {
        "charge_rows": len(rows),
        "mulliken_sum": mull_sum,
        "loewdin_sum": loew_sum,
        "n_rows": n_rows,
        "o_rows": o_rows,
    }


def get_table9_value():
    if not TABLE9.exists():
        return None

    with TABLE9.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("Compound_ID") == MOL_ID:
                return r
    return None


def add(rows, field, value, status=None, note=""):
    rows.append({
        "Field": field,
        "Value": value,
        "Status": status if status is not None else "",
        "Note": note,
    })


def main():
    rows = []

    opt_inp_text = read_text(OPT_INP)
    sp_inp_text = read_text(SP_INP)

    opt_out = parse_orca_out(OPT_OUT)
    sp_out = parse_orca_out(SP_OUT)

    opt_charge, opt_mult, opt_xyzline = parse_charge_mult(opt_inp_text)
    sp_charge, sp_mult, sp_xyzline = parse_charge_mult(sp_inp_text)

    opt_kw = keyword_check(opt_inp_text, EXPECTED_KEYWORDS_OPT)
    sp_kw = keyword_check(sp_inp_text, EXPECTED_KEYWORDS_SP)

    start_xyz_count = count_xyz_atoms(START_XYZ)
    opt_xyz_count = count_xyz_atoms(OPT_XYZ)

    charges = parse_charges_csv(CHARGE_CSV)
    table9 = get_table9_value()

    # File existence
    for p in [
        OPT_INP, SP_INP, OPT_OUT, SP_OUT, START_XYZ, OPT_XYZ,
        SP_GBW_INPUT, SP_DENSITIES_INPUT, DENSITY_CUBE, ESP_CUBE,
        MEP_PNG, CHARGE_CSV, SP_SUMMARY_CSV
    ]:
        add(rows, f"exists::{p.name}", p.exists(), "PASS" if p.exists() else "CHECK", str(p))

    # Inputs
    add(rows, "Opt route", route_line(opt_inp_text), "PASS" if all(opt_kw.values()) else "CHECK")
    add(rows, "SP route", route_line(sp_inp_text), "PASS" if all(sp_kw.values()) else "CHECK")

    for k, v in opt_kw.items():
        add(rows, f"Opt keyword {k}", v, "PASS" if v else "CHECK")

    for k, v in sp_kw.items():
        add(rows, f"SP keyword {k}", v, "PASS" if v else "CHECK")

    add(rows, "Opt charge/multiplicity line", opt_xyzline, "PASS" if opt_charge == "0" and opt_mult == "1" else "CHECK")
    add(rows, "SP charge/multiplicity line", sp_xyzline, "PASS" if sp_charge == "0" and sp_mult == "1" else "CHECK")

    # Output integrity
    add(rows, "Opt normal termination", opt_out["normal"], "PASS" if opt_out["normal"] else "CHECK")
    add(rows, "Opt convergence", opt_out["opt_converged"], "PASS" if opt_out["opt_converged"] else "CHECK")
    add(rows, "Opt frequency count", opt_out["frequency_count"], "PASS" if opt_out["frequency_count"] == 132 else "CHECK")
    add(rows, "Opt min frequency", opt_out["min_frequency"], "PASS" if opt_out["min_frequency"] is not None and opt_out["min_frequency"] > -20 else "CHECK")
    add(rows, "Opt imaginary frequencies < -20 cm-1", opt_out["imaginary_lt_minus20_values"], "PASS" if opt_out["imaginary_lt_minus20_count"] == 0 else "CHECK")
    add(rows, "Opt final energy count", opt_out["final_energy_count"], "PASS" if opt_out["final_energy_count"] > 0 else "CHECK")
    add(rows, "Opt last final energy", opt_out["last_final_energy_line"], "PASS" if opt_out["last_final_energy_line"] else "CHECK")

    add(rows, "SP normal termination", sp_out["normal"], "PASS" if sp_out["normal"] else "CHECK")
    add(rows, "SP orbital row count", sp_out["orbital_row_count"], "PASS" if sp_out["orbital_row_count"] > 0 else "CHECK")
    add(rows, "SP HOMO eV", sp_out["HOMO_eV"], "PASS" if sp_out["HOMO_eV"] is not None else "CHECK")
    add(rows, "SP LUMO eV", sp_out["LUMO_eV"], "PASS" if sp_out["LUMO_eV"] is not None else "CHECK")
    add(rows, "SP gap eV", sp_out["Gap_eV"], "PASS" if sp_out["Gap_eV"] is not None and sp_out["Gap_eV"] > 0 else "CHECK")
    add(rows, "SP dipole D", sp_out["Dipole_D"], "PASS" if sp_out["Dipole_D"] is not None else "CHECK")
    add(rows, "SP final energy count", sp_out["final_energy_count"], "PASS" if sp_out["final_energy_count"] > 0 else "CHECK")
    add(rows, "SP last final energy", sp_out["last_final_energy_line"], "PASS" if sp_out["last_final_energy_line"] else "CHECK")

    # Dipole sanity
    if sp_out["Dipole_D"] is not None:
        if sp_out["Dipole_D"] < 0:
            dip_status = "CHECK"
            dip_note = "Negative dipole magnitude impossible"
        elif sp_out["Dipole_D"] > 20:
            dip_status = "CHECK"
            dip_note = "Very high; inspect"
        elif sp_out["Dipole_D"] > 8:
            dip_status = "PASS"
            dip_note = "High but chemically plausible for strongly polar substituted molecule"
        else:
            dip_status = "PASS"
            dip_note = "Within ordinary range"
        add(rows, "Dipole plausibility check", sp_out["Dipole_D"], dip_status, dip_note)

    # XYZ atom counts
    add(rows, "Start XYZ atom count", start_xyz_count, "PASS" if start_xyz_count == (44, 44) else "CHECK")
    add(rows, "Opt XYZ atom count", opt_xyz_count, "PASS" if opt_xyz_count == (44, 44) else "CHECK")

    # Charges
    if charges:
        add(rows, "Charge rows", charges["charge_rows"], "PASS" if charges["charge_rows"] == 44 else "CHECK")
        add(rows, "Mulliken charge sum", charges["mulliken_sum"], "PASS" if abs(charges["mulliken_sum"]) < 1e-3 else "CHECK")
        add(rows, "Loewdin charge sum", charges["loewdin_sum"], "PASS" if abs(charges["loewdin_sum"]) < 1e-3 else "CHECK")
        add(rows, "Nitrogen charge rows", charges["n_rows"], "PASS" if len(charges["n_rows"]) == 2 else "CHECK")
        add(rows, "Oxygen charge rows", charges["o_rows"], "PASS" if len(charges["o_rows"]) == 4 else "CHECK")
    else:
        add(rows, "Charge CSV parsed", False, "CHECK")

    # Table9 consistency
    if table9:
        try:
            t_homo = float(table9["HOMO_eV"])
            t_lumo = float(table9["LUMO_eV"])
            t_gap = float(table9["Gap_eV"])
            t_dip = float(table9["Dipole_D"])

            add(rows, "Table9 HOMO agrees", abs(t_homo - sp_out["HOMO_eV"]) < 1e-4, "PASS" if abs(t_homo - sp_out["HOMO_eV"]) < 1e-4 else "CHECK", f"Table9={t_homo}, output={sp_out['HOMO_eV']}")
            add(rows, "Table9 LUMO agrees", abs(t_lumo - sp_out["LUMO_eV"]) < 1e-4, "PASS" if abs(t_lumo - sp_out["LUMO_eV"]) < 1e-4 else "CHECK", f"Table9={t_lumo}, output={sp_out['LUMO_eV']}")
            add(rows, "Table9 gap agrees", abs(t_gap - sp_out["Gap_eV"]) < 1e-4, "PASS" if abs(t_gap - sp_out["Gap_eV"]) < 1e-4 else "CHECK", f"Table9={t_gap}, output={sp_out['Gap_eV']}")
            add(rows, "Table9 dipole agrees", abs(t_dip - sp_out["Dipole_D"]) < 1e-3, "PASS" if abs(t_dip - sp_out["Dipole_D"]) < 1e-3 else "CHECK", f"Table9={t_dip}, output={sp_out['Dipole_D']}")
        except Exception as e:
            add(rows, "Table9 consistency parse", str(e), "CHECK")
    else:
        add(rows, "Table9 row found", False, "CHECK")

    # Overall
    hard_checks = [
        r for r in rows
        if r["Field"].startswith("exists::")
        or r["Field"] in [
            "Opt route", "SP route",
            "Opt charge/multiplicity line", "SP charge/multiplicity line",
            "Opt normal termination", "Opt convergence",
            "Opt imaginary frequencies < -20 cm-1",
            "SP normal termination", "SP HOMO eV", "SP LUMO eV",
            "SP gap eV", "SP dipole D",
            "Start XYZ atom count", "Opt XYZ atom count",
            "Charge rows", "Mulliken charge sum", "Loewdin charge sum",
            "Nitrogen charge rows", "Oxygen charge rows",
        ]
    ]

    hard_pass = all(r["Status"] == "PASS" for r in hard_checks)

    add(rows, "OVERALL_FULL_AUDIT", "PASS" if hard_pass else "CHECK", "PASS" if hard_pass else "CHECK")

    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Field", "Value", "Status", "Note"])
        writer.writeheader()
        writer.writerows(rows)

    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"{MOL_ID} FULL AUDIT REPORT\n")
        f.write("=" * 100 + "\n\n")

        for r in rows:
            f.write(f"{r['Status']:5s} | {r['Field']}: {r['Value']}")
            if r["Note"]:
                f.write(f" | {r['Note']}")
            f.write("\n")

        f.write("\n\nDIPOLE BLOCK EXCERPT FROM SP OUTPUT\n")
        f.write("-" * 100 + "\n")
        f.write(sp_out["dipole_block_excerpt"] + "\n")

    print(f"\n{MOL_ID} full audit completed.")
    print(REPORT_TXT)
    print(REPORT_CSV)
    print("\nSummary:")
    for r in rows:
        if r["Field"] in [
            "Opt normal termination", "Opt convergence",
            "Opt frequency count", "Opt min frequency",
            "Opt imaginary frequencies < -20 cm-1",
            "SP normal termination", "SP HOMO eV", "SP LUMO eV",
            "SP gap eV", "SP dipole D",
            "Dipole plausibility check",
            "Mulliken charge sum", "Loewdin charge sum",
            "OVERALL_FULL_AUDIT",
        ]:
            print(f"{r['Status']:5s} | {r['Field']}: {r['Value']} | {r['Note']}")


if __name__ == "__main__":
    main()