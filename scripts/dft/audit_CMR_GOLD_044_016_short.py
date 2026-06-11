from pathlib import Path
import re
import csv

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
TABLE9 = PROJECT / "tables" / "Table9_DFT_panel_verified.csv"
OUTDIR = PROJECT / "audit_existing"
OUTDIR.mkdir(parents=True, exist_ok=True)

MOLECULES = {
    "CMR_GOLD_044": {
        "expected_atoms": 25,
        "expected_n": 2,
        "expected_o": 3,
        "expected_freq_count": 75,
        "opt_out": "CMR_GOLD_044_optfreq.out",
        "sp_out": "CMR_GOLD_044_sp.out",
    },
    "CMR_GOLD_016": {
        "expected_atoms": 46,
        "expected_n": 0,
        "expected_o": 8,
        "expected_freq_count": 138,
        "opt_out": "CMR_GOLD_016_optfreq.out",
        "sp_out": "CMR_GOLD_016_sp.out",
    },
}

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


def route_line(inp_text):
    for line in inp_text.splitlines():
        s = line.strip()
        if s.startswith("!"):
            return s
    return ""


def parse_charge_mult(inp_text):
    for line in inp_text.splitlines():
        s = line.strip()
        if s.startswith("* xyzfile"):
            parts = s.split()
            if len(parts) >= 4:
                return parts[2], parts[3], s
    return None, None, ""


def parse_orca_out(path):
    text = read_text(path)
    lines = text.splitlines()

    normal = "ORCA TERMINATED NORMALLY" in text
    converged = (
        "THE OPTIMIZATION HAS CONVERGED" in text
        or "OPTIMIZATION RUN DONE" in text
        or "HURRAY" in text
    )

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

    imag_lt_minus20 = [f for f in freqs if f < -20.0]

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
                    hartree = float(parts[2])
                    ev = float(parts[3])
                    orbital_rows.append((idx, occ, hartree, ev))
                except ValueError:
                    pass

    homo = None
    lumo = None

    for row in orbital_rows:
        idx, occ, hartree, ev = row
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

    final_energy_count = sum(1 for line in lines if "FINAL SINGLE POINT ENERGY" in line)

    return {
        "normal": normal,
        "converged": converged,
        "frequency_count": len(freqs),
        "min_frequency": min(freqs) if freqs else None,
        "imag_lt_minus20": imag_lt_minus20,
        "orbital_rows": len(orbital_rows),
        "HOMO_eV": homo[3] if homo else None,
        "LUMO_eV": lumo[3] if lumo else None,
        "Gap_eV": (lumo[3] - homo[3]) if homo and lumo else None,
        "Dipole_D": dipole,
        "final_energy_count": final_energy_count,
    }


def count_xyz_atoms(path):
    if not path.exists():
        return None, None

    lines = path.read_text(errors="ignore").splitlines()

    try:
        header_n = int(lines[0].strip())
    except Exception:
        header_n = None

    coord_n = sum(1 for line in lines[2:] if len(line.split()) >= 4)
    return header_n, coord_n


def parse_charges(path):
    if not path.exists():
        return None

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    mull_sum = 0.0
    loew_sum = 0.0
    n_count = 0
    o_count = 0

    for r in rows:
        el = r.get("Element", "")

        try:
            mull_sum += float(r.get("Mulliken_charge", 0))
        except Exception:
            pass

        try:
            loew_sum += float(r.get("Loewdin_charge", 0))
        except Exception:
            pass

        if el == "N":
            n_count += 1
        if el == "O":
            o_count += 1

    return {
        "rows": len(rows),
        "mulliken_sum": mull_sum,
        "loewdin_sum": loew_sum,
        "n_count": n_count,
        "o_count": o_count,
    }


def load_table9():
    if not TABLE9.exists():
        return {}

    data = {}
    with TABLE9.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            data[r["Compound_ID"]] = r
    return data


def add(result_rows, mol_id, field, value, status, note=""):
    result_rows.append({
        "Compound_ID": mol_id,
        "Field": field,
        "Value": value,
        "Status": status,
        "Note": note,
    })


def audit_molecule(mol_id, cfg, table9):
    result_rows = []

    mol_dir = PROJECT / "dft" / "molecules" / mol_id

    opt_inp = mol_dir / "input" / f"{mol_id}_optfreq.inp"
    sp_inp = mol_dir / "input" / f"{mol_id}_sp.inp"

    opt_out = mol_dir / "output" / cfg["opt_out"]
    sp_out = mol_dir / "output" / cfg["sp_out"]

    start_xyz = mol_dir / "input" / f"{mol_id}_start.xyz"
    opt_xyz = mol_dir / "geometry" / f"{mol_id}_opt.xyz"

    charge_csv = mol_dir / "extracted_data" / f"{mol_id}_charges_mulliken_loewdin.csv"
    sp_summary = mol_dir / "extracted_data" / f"{mol_id}_sp_summary.csv"

    opt_inp_text = read_text(opt_inp)
    sp_inp_text = read_text(sp_inp)

    opt = parse_orca_out(opt_out)
    sp = parse_orca_out(sp_out)

    # File existence
    for p in [opt_inp, sp_inp, opt_out, sp_out, start_xyz, opt_xyz, charge_csv, sp_summary]:
        add(result_rows, mol_id, f"exists::{p.name}", p.exists(), "PASS" if p.exists() else "CHECK", str(p))

    # Input method
    opt_route = route_line(opt_inp_text)
    sp_route = route_line(sp_inp_text)

    add(result_rows, mol_id, "Opt route", opt_route, "PASS" if all(k in opt_inp_text for k in EXPECTED_KEYWORDS_OPT) else "CHECK")
    add(result_rows, mol_id, "SP route", sp_route, "PASS" if all(k in sp_inp_text for k in EXPECTED_KEYWORDS_SP) else "CHECK")

    opt_charge, opt_mult, opt_xyzline = parse_charge_mult(opt_inp_text)
    sp_charge, sp_mult, sp_xyzline = parse_charge_mult(sp_inp_text)

    add(result_rows, mol_id, "Opt charge/multiplicity", opt_xyzline, "PASS" if opt_charge == "0" and opt_mult == "1" else "CHECK")
    add(result_rows, mol_id, "SP charge/multiplicity", sp_xyzline, "PASS" if sp_charge == "0" and sp_mult == "1" else "CHECK")

    # Opt/Freq
    add(result_rows, mol_id, "Opt normal termination", opt["normal"], "PASS" if opt["normal"] else "CHECK")
    add(result_rows, mol_id, "Opt convergence", opt["converged"], "PASS" if opt["converged"] else "CHECK")
    add(
        result_rows,
        mol_id,
        "Opt frequency count",
        opt["frequency_count"],
        "PASS" if opt["frequency_count"] == cfg["expected_freq_count"] else "CHECK",
        f"expected={cfg['expected_freq_count']}"
    )
    add(result_rows, mol_id, "Opt min frequency", opt["min_frequency"], "PASS" if opt["min_frequency"] is not None and opt["min_frequency"] > -20 else "CHECK")
    add(result_rows, mol_id, "Opt imaginary frequencies < -20 cm-1", opt["imag_lt_minus20"], "PASS" if len(opt["imag_lt_minus20"]) == 0 else "CHECK")
    add(result_rows, mol_id, "Opt final energy count", opt["final_energy_count"], "PASS" if opt["final_energy_count"] > 0 else "CHECK")

    # SP
    add(result_rows, mol_id, "SP normal termination", sp["normal"], "PASS" if sp["normal"] else "CHECK")
    add(result_rows, mol_id, "SP orbital row count", sp["orbital_rows"], "PASS" if sp["orbital_rows"] > 0 else "CHECK")
    add(result_rows, mol_id, "SP HOMO eV", sp["HOMO_eV"], "PASS" if sp["HOMO_eV"] is not None else "CHECK")
    add(result_rows, mol_id, "SP LUMO eV", sp["LUMO_eV"], "PASS" if sp["LUMO_eV"] is not None else "CHECK")
    add(result_rows, mol_id, "SP gap eV", sp["Gap_eV"], "PASS" if sp["Gap_eV"] is not None and sp["Gap_eV"] > 0 else "CHECK")
    add(result_rows, mol_id, "SP dipole D", sp["Dipole_D"], "PASS" if sp["Dipole_D"] is not None else "CHECK")
    add(result_rows, mol_id, "SP final energy count", sp["final_energy_count"], "PASS" if sp["final_energy_count"] > 0 else "CHECK")

    # XYZ atom counts
    start_header, start_coord = count_xyz_atoms(start_xyz)
    opt_header, opt_coord = count_xyz_atoms(opt_xyz)

    add(
        result_rows,
        mol_id,
        "Start XYZ atom count",
        (start_header, start_coord),
        "PASS" if (start_header, start_coord) == (cfg["expected_atoms"], cfg["expected_atoms"]) else "CHECK",
    )
    add(
        result_rows,
        mol_id,
        "Opt XYZ atom count",
        (opt_header, opt_coord),
        "PASS" if (opt_header, opt_coord) == (cfg["expected_atoms"], cfg["expected_atoms"]) else "CHECK",
    )

    # Charges
    charges = parse_charges(charge_csv)
    if charges:
        add(result_rows, mol_id, "Charge rows", charges["rows"], "PASS" if charges["rows"] == cfg["expected_atoms"] else "CHECK")
        add(result_rows, mol_id, "Mulliken charge sum", charges["mulliken_sum"], "PASS" if abs(charges["mulliken_sum"]) < 1e-3 else "CHECK")
        add(result_rows, mol_id, "Loewdin charge sum", charges["loewdin_sum"], "PASS" if abs(charges["loewdin_sum"]) < 1e-3 else "CHECK")
        add(result_rows, mol_id, "N atom count in charges", charges["n_count"], "PASS" if charges["n_count"] == cfg["expected_n"] else "CHECK")
        add(result_rows, mol_id, "O atom count in charges", charges["o_count"], "PASS" if charges["o_count"] == cfg["expected_o"] else "CHECK")
    else:
        add(result_rows, mol_id, "Charges parsed", False, "CHECK")

    # Table9 consistency
    if mol_id in table9:
        row = table9[mol_id]

        checks = [
            ("HOMO_eV", sp["HOMO_eV"], 1e-4),
            ("LUMO_eV", sp["LUMO_eV"], 1e-4),
            ("Gap_eV", sp["Gap_eV"], 1e-4),
            ("Dipole_D", sp["Dipole_D"], 1e-3),
        ]

        for col, out_value, tol in checks:
            try:
                table_value = float(row[col])
                diff = abs(table_value - out_value)
                add(
                    result_rows,
                    mol_id,
                    f"Table9 {col} agrees",
                    diff,
                    "PASS" if diff <= tol else "CHECK",
                    f"Table9={table_value}, output={out_value}, tol={tol}"
                )
            except Exception as e:
                add(result_rows, mol_id, f"Table9 {col} parse", str(e), "CHECK")
    else:
        add(result_rows, mol_id, "Table9 row found", False, "CHECK")

    # Overall hard checks
    hard_fail = [
        r for r in result_rows
        if r["Status"] != "PASS"
        and (
            r["Field"].startswith("exists::")
            or r["Field"] in {
                "Opt route",
                "SP route",
                "Opt charge/multiplicity",
                "SP charge/multiplicity",
                "Opt normal termination",
                "Opt convergence",
                "Opt imaginary frequencies < -20 cm-1",
                "SP normal termination",
                "SP HOMO eV",
                "SP LUMO eV",
                "SP gap eV",
                "SP dipole D",
                "Start XYZ atom count",
                "Opt XYZ atom count",
                "Charge rows",
                "Mulliken charge sum",
                "Loewdin charge sum",
                "N atom count in charges",
                "O atom count in charges",
            }
        )
    ]

    add(result_rows, mol_id, "OVERALL_SHORT_AUDIT", "PASS" if not hard_fail else "CHECK", "PASS" if not hard_fail else "CHECK")

    return result_rows


def main():
    table9 = load_table9()
    all_rows = []

    for mol_id, cfg in MOLECULES.items():
        rows = audit_molecule(mol_id, cfg, table9)
        all_rows.extend(rows)

    out_csv = OUTDIR / "CMR_GOLD_044_016_short_method_audit.csv"
    out_txt = OUTDIR / "CMR_GOLD_044_016_short_method_audit.txt"

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound_ID", "Field", "Value", "Status", "Note"])
        writer.writeheader()
        writer.writerows(all_rows)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("CMR_GOLD_044 and CMR_GOLD_016 short method audit\n")
        f.write("=" * 100 + "\n\n")

        for mol_id in MOLECULES:
            f.write(f"\n{mol_id}\n")
            f.write("-" * 100 + "\n")

            for r in all_rows:
                if r["Compound_ID"] == mol_id:
                    f.write(f"{r['Status']:5s} | {r['Field']}: {r['Value']}")
                    if r["Note"]:
                        f.write(f" | {r['Note']}")
                    f.write("\n")

    print("\nAudit tamamlandı.")
    print(out_txt)
    print(out_csv)

    print("\nSummary:")
    for mol_id in MOLECULES:
        print(f"\n{mol_id}")
        for r in all_rows:
            if r["Compound_ID"] == mol_id and r["Field"] in [
                "Opt normal termination",
                "Opt convergence",
                "Opt frequency count",
                "Opt min frequency",
                "Opt imaginary frequencies < -20 cm-1",
                "SP normal termination",
                "SP HOMO eV",
                "SP LUMO eV",
                "SP gap eV",
                "SP dipole D",
                "Mulliken charge sum",
                "Loewdin charge sum",
                "OVERALL_SHORT_AUDIT",
            ]:
                print(f"{r['Status']:5s} | {r['Field']}: {r['Value']} | {r['Note']}")


if __name__ == "__main__":
    main()