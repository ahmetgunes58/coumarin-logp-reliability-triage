from pathlib import Path
import csv
import re

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")

TABLE_DIR = PROJECT / "tables"
AUDIT_DIR = PROJECT / "audit_existing"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

COMPOUNDS = [
    {
        "Compound_ID": "CMR_GOLD_055",
        "Role": "N-free reference / parent core",
        "N_count": 0,
        "logP_exp": 1.39,
        "Consensus_logP": 1.82,
        "Delta_logP": -0.43,
        "opt_out": "CMR_GOLD_055_optfreq_existing.out",
        "sp_out": "CMR_GOLD_055_sp_existing.out",
    },
    {
        "Compound_ID": "CMR_GOLD_043",
        "Role": "FM0 oxadiazole accurate #1",
        "N_count": 2,
        "logP_exp": 1.95,
        "Consensus_logP": 1.78,
        "Delta_logP": 0.17,
        "opt_out": "CMR_GOLD_043_optfreq_existing.out",
        "sp_out": "CMR_GOLD_043_sp_existing.out",
    },
    {
        "Compound_ID": "CMR_GOLD_044",
        "Role": "FM0 oxadiazole accurate #2",
        "N_count": 2,
        "logP_exp": 2.12,
        "Consensus_logP": 2.15,
        "Delta_logP": -0.03,
        "opt_out": "CMR_GOLD_044_optfreq.out",
        "sp_out": "CMR_GOLD_044_sp.out",
    },
    {
        "Compound_ID": "CMR_GOLD_029",
        "Role": "FM1 N=2 conjugated failure",
        "N_count": 2,
        "logP_exp": 0.55,
        "Consensus_logP": 2.59,
        "Delta_logP": -2.04,
        "opt_out": "CMR_GOLD_029_optfreq.out",
        "sp_out": "CMR_GOLD_029_sp.out",
    },
    {
        "Compound_ID": "CMR_GOLD_058",
        "Role": "FM1 extreme D-pi-A failure",
        "N_count": 2,
        "logP_exp": 0.97,
        "Consensus_logP": 6.16,
        "Delta_logP": -5.19,
        "opt_out": "CMR_GOLD_058_optfreq_existing.out",
        "sp_out": "CMR_GOLD_058_sp_existing.out",
    },
    {
        "Compound_ID": "CMR_GOLD_079",
        "Role": "FM3 high-N cancellation",
        "N_count": 4,
        "logP_exp": 2.02,
        "Consensus_logP": 1.97,
        "Delta_logP": 0.05,
        "opt_out": "CMR_GOLD_079_optfreq_existing.out",
        "sp_out": "CMR_GOLD_079_sp_existing.out",
    },
    {
        "Compound_ID": "CMR_GOLD_016",
        "Role": "FM4 dimeric coumarin opposite-bias",
        "N_count": 0,
        "logP_exp": 4.93,
        "Consensus_logP": 2.66,
        "Delta_logP": 2.27,
        "opt_out": "CMR_GOLD_016_optfreq.out",
        "sp_out": "CMR_GOLD_016_sp.out",
    },
]


def parse_orca_out(path):
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    normal = "ORCA TERMINATED NORMALLY" in text
    converged = (
        "THE OPTIMIZATION HAS CONVERGED" in text
        or "HURRAY" in text
        or "OPTIMIZATION RUN DONE" in text
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

    route_hits = []
    for key in ["B3LYP", "D3BJ", "def2-SVP", "CPCM", "TightSCF", "RIJCOSX", "def2/J"]:
        route_hits.append((key, key in text))

    return {
        "normal": normal,
        "converged": converged,
        "frequency_count": len(freqs),
        "imaginary_lt_minus20": imag,
        "HOMO_eV": homo[3] if homo else None,
        "LUMO_eV": lumo[3] if lumo else None,
        "Gap_eV": (lumo[3] - homo[3]) if homo and lumo else None,
        "Dipole_D": dipole,
        "method_keywords": route_hits,
    }


def method_keyword_ok(parsed):
    return all(v for k, v in parsed["method_keywords"])


def main():
    table_rows = []
    audit_rows = []

    for c in COMPOUNDS:
        mol_id = c["Compound_ID"]
        mol_dir = PROJECT / "dft" / "molecules" / mol_id
        out_dir = mol_dir / "output"

        opt_path = out_dir / c["opt_out"]
        sp_path = out_dir / c["sp_out"]

        if not opt_path.exists():
            raise FileNotFoundError(f"Opt/Freq output yok: {opt_path}")

        if not sp_path.exists():
            raise FileNotFoundError(f"SP output yok: {sp_path}")

        opt = parse_orca_out(opt_path)
        sp = parse_orca_out(sp_path)

        opt_ok = (
            opt["normal"]
            and opt["converged"]
            and opt["frequency_count"] > 0
            and len(opt["imaginary_lt_minus20"]) == 0
        )

        sp_ok = (
            sp["normal"]
            and sp["HOMO_eV"] is not None
            and sp["LUMO_eV"] is not None
            and sp["Gap_eV"] is not None
            and sp["Dipole_D"] is not None
        )

        audit_status = "PASS" if opt_ok and sp_ok else "CHECK"

        table_rows.append({
            "Compound_ID": mol_id,
            "Role": c["Role"],
            "N_count": c["N_count"],
            "logP_exp": c["logP_exp"],
            "Consensus_logP": c["Consensus_logP"],
            "Delta_logP_Exp_minus_Pred": c["Delta_logP"],
            "Abs_Delta_logP": abs(c["Delta_logP"]),
            "HOMO_eV": round(sp["HOMO_eV"], 4),
            "LUMO_eV": round(sp["LUMO_eV"], 4),
            "Gap_eV": round(sp["Gap_eV"], 4),
            "Dipole_D": round(sp["Dipole_D"], 4),
            "OptFreq_status": "PASS" if opt_ok else "CHECK",
            "SP_status": "PASS" if sp_ok else "CHECK",
            "Overall_status": audit_status,
            "OptFreq_output": str(opt_path),
            "SP_output": str(sp_path),
        })

        audit_rows.append({
            "Compound_ID": mol_id,
            "OptFreq_output": str(opt_path),
            "SP_output": str(sp_path),
            "Opt_normal": opt["normal"],
            "Opt_converged": opt["converged"],
            "Opt_frequency_count": opt["frequency_count"],
            "Opt_imaginary_lt_minus20": ";".join(str(x) for x in opt["imaginary_lt_minus20"]),
            "SP_normal": sp["normal"],
            "SP_HOMO_eV": sp["HOMO_eV"],
            "SP_LUMO_eV": sp["LUMO_eV"],
            "SP_Gap_eV": sp["Gap_eV"],
            "SP_Dipole_D": sp["Dipole_D"],
            "Overall_status": audit_status,
        })

    table_csv = TABLE_DIR / "Table9_DFT_panel_verified.csv"
    audit_csv = AUDIT_DIR / "DFT_panel_7compound_method_audit.csv"
    table_txt = TABLE_DIR / "Table9_DFT_panel_verified.txt"

    with table_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    with audit_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    with table_txt.open("w", encoding="utf-8") as f:
        f.write("Verified Table 9 DFT panel\n")
        f.write("=" * 120 + "\n\n")
        for r in table_rows:
            f.write(
                f"{r['Compound_ID']:12s} | {r['Role']:38s} | "
                f"N={r['N_count']} | Exp={r['logP_exp']} | Cons={r['Consensus_logP']} | "
                f"Delta={r['Delta_logP_Exp_minus_Pred']} | "
                f"HOMO={r['HOMO_eV']} | LUMO={r['LUMO_eV']} | "
                f"Gap={r['Gap_eV']} | Dipole={r['Dipole_D']} | "
                f"Status={r['Overall_status']}\n"
            )

    print("\nVerified Table 9 written:")
    print(table_csv)
    print(table_txt)
    print("\nAudit file written:")
    print(audit_csv)

    print("\n=== VERIFIED TABLE 9 ===")
    for r in table_rows:
        print(
            f"{r['Compound_ID']:12s} | "
            f"HOMO {r['HOMO_eV']:>8.4f} | "
            f"LUMO {r['LUMO_eV']:>8.4f} | "
            f"Gap {r['Gap_eV']:>7.4f} | "
            f"Dipole {r['Dipole_D']:>7.4f} | "
            f"{r['Overall_status']}"
        )


if __name__ == "__main__":
    main()