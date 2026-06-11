# -*- coding: utf-8 -*-
"""
Build 10-compound DFT panel summary after adding three FM2 anchors.

Existing 7-compound panel:
- CMR_GOLD_016
- CMR_GOLD_029
- CMR_GOLD_044
- CMR_GOLD_043
- CMR_GOLD_055
- CMR_GOLD_058
- CMR_GOLD_079

Added FM2 anchors:
- CMR_GOLD_090
- CMR_GOLD_020
- CMR_GOLD_092

Outputs:
- Dataset_S34_DFT_panel_10_summary.csv
- Dataset_S34_DFT_panel_10_status_report.txt
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DATA = ROOT / "data" / "processed"
DFT = ROOT / "dft" / "molecules"

PANEL_IDS = [
    "CMR_GOLD_016",
    "CMR_GOLD_029",
    "CMR_GOLD_044",
    "CMR_GOLD_043",
    "CMR_GOLD_055",
    "CMR_GOLD_058",
    "CMR_GOLD_079",
    "CMR_GOLD_090",
    "CMR_GOLD_020",
    "CMR_GOLD_092",
]

DFT_ROLES = {
    "CMR_GOLD_016": "FM4 N-free π-extended boundary / underestimation case",
    "CMR_GOLD_029": "FM1 polar overestimation / high-dipole case",
    "CMR_GOLD_044": "Serviceable compact N=2 control with low consensus error",
    "CMR_GOLD_043": "Serviceable compact N=2 control with low consensus error",
    "CMR_GOLD_055": "FM0 N-free serviceable reference anchor",
    "CMR_GOLD_058": "FM1 extreme donor-acceptor-conjugated severe overestimation case",
    "CMR_GOLD_079": "FM3 mixed high-N regime anchor",
    "CMR_GOLD_090": "FM2 representative class-central anchor",
    "CMR_GOLD_020": "FM2 N=3 high-Spread4 conjugated anchor",
    "CMR_GOLD_092": "FM2 severe high-risk / high-disagreement anchor",
}

DATASET = DATA / "Dataset_S1_benchmark_dataset.csv"
OUT_CSV = DATA / "Dataset_S34_DFT_panel_10_summary.csv"
OUT_REPORT = DATA / "Dataset_S34_DFT_panel_10_status_report.txt"


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def terminated(text):
    return "ORCA TERMINATED NORMALLY" in text


def final_energy(text):
    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    return float(matches[-1]) if matches else np.nan


def parse_frequencies(text):
    freqs = []
    pat = re.compile(r"^\s*\d+:\s*(-?\d+\.\d+)\s*cm\*\*-1")
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            freqs.append(float(m.group(1)))
    freqs = sorted(freqs)
    neg = [f for f in freqs if f < -20.0]
    small_neg = [f for f in freqs if -20.0 <= f < 0.0]
    positive = [f for f in freqs if f > 0]
    return {
        "n_frequencies": len(freqs),
        "n_imaginary_lt_minus20": len(neg),
        "imaginary_lt_minus20": ";".join(f"{x:.2f}" for x in neg),
        "small_negative_modes": ";".join(f"{x:.2f}" for x in small_neg),
        "lowest_positive_frequency_cm-1": positive[0] if positive else np.nan,
    }


def parse_orbitals_from_sp(text):
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if "ORBITAL ENERGIES" in line]
    if not starts:
        return {
            "HOMO_orbital": np.nan,
            "HOMO_eV": np.nan,
            "LUMO_orbital": np.nan,
            "LUMO_eV": np.nan,
            "Gap_eV": np.nan,
        }

    start = starts[-1]
    block = []
    for line in lines[start + 1:]:
        upper = line.upper()
        if "MOLECULAR ORBITALS" in upper:
            break
        if "MULLIKEN" in upper or "LOEWDIN" in upper or "DIPOLE MOMENT" in upper:
            break
        block.append(line)

    rows = []
    for line in block:
        parts = line.split()
        if len(parts) >= 4:
            try:
                rows.append({
                    "orbital_no": int(parts[0]),
                    "occupation": float(parts[1]),
                    "energy_Eh": float(parts[2]),
                    "energy_eV": float(parts[3]),
                })
            except Exception:
                pass

    occupied = [r for r in rows if r["occupation"] > 0.0]
    virtual = [r for r in rows if r["occupation"] == 0.0]

    if not occupied or not virtual:
        return {
            "HOMO_orbital": np.nan,
            "HOMO_eV": np.nan,
            "LUMO_orbital": np.nan,
            "LUMO_eV": np.nan,
            "Gap_eV": np.nan,
        }

    homo = occupied[-1]
    lumo = virtual[0]
    return {
        "HOMO_orbital": homo["orbital_no"],
        "HOMO_eV": homo["energy_eV"],
        "LUMO_orbital": lumo["orbital_no"],
        "LUMO_eV": lumo["energy_eV"],
        "Gap_eV": lumo["energy_eV"] - homo["energy_eV"],
    }


def parse_dipole(text):
    matches = re.findall(r"Magnitude\s+\(Debye\)\s*:\s*([+-]?\d+\.\d+)", text)
    return float(matches[-1]) if matches else np.nan


def find_file(mol_dir, patterns):
    for pattern in patterns:
        hits = list(mol_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {DATASET}")

    df = pd.read_csv(DATASET)

    meta_cols = [
        "Compound_ID", "SMILES", "FM", "N_count", "logP_exp", "Consensus",
        "delta_Consensus", "TPSA", "MW", "Coumarin_Type"
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]

    meta = df[df["Compound_ID"].isin(PANEL_IDS)][meta_cols].copy()

    rows = []

    for cid in PANEL_IDS:
        mol_dir = DFT / cid

        opt_out = mol_dir / "output" / f"{cid}_optfreq.out"
        sp_out = mol_dir / "output" / f"{cid}_sp.out"
        opt_xyz = mol_dir / "geometry" / f"{cid}_opt.xyz"

        charge_all = mol_dir / "extracted_data" / f"{cid}_charges_mulliken_loewdin.csv"
        charge_heavy = mol_dir / "extracted_data" / f"{cid}_heavy_atom_charges.csv"
        frontier_csv = mol_dir / "extracted_data" / f"{cid}_frontier_orbitals.csv"

        opt_text = read_text(opt_out)
        sp_text = read_text(sp_out)

        freq = parse_frequencies(opt_text)
        orbital = parse_orbitals_from_sp(sp_text)

        row = {
            "Compound_ID": cid,
            "DFT_role": DFT_ROLES.get(cid, ""),
            "mol_dir_exists": mol_dir.exists(),
            "optfreq_out_exists": opt_out.exists(),
            "sp_out_exists": sp_out.exists(),
            "opt_xyz_exists": opt_xyz.exists(),
            "charge_all_exists": charge_all.exists(),
            "charge_heavy_exists": charge_heavy.exists(),
            "frontier_csv_exists": frontier_csv.exists(),
            "optfreq_terminated_normally": terminated(opt_text),
            "sp_terminated_normally": terminated(sp_text),
            "optfreq_final_energy_Eh": final_energy(opt_text),
            "sp_final_energy_Eh": final_energy(sp_text),
            "dipole_D": parse_dipole(sp_text),
        }
        row.update(freq)
        row.update(orbital)
        rows.append(row)

    status = pd.DataFrame(rows)
    merged = pd.DataFrame({"Compound_ID": PANEL_IDS}).merge(meta, on="Compound_ID", how="left").merge(status, on="Compound_ID", how="left")

    for c in ["HOMO_eV", "LUMO_eV", "Gap_eV", "dipole_D"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").round(4)

    for c in ["optfreq_final_energy_Eh", "sp_final_energy_Eh"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").round(12)

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("10-compound DFT panel status report\n")
        f.write("=" * 80 + "\n\n")
        f.write("Protocol: B3LYP-D3BJ/def2-SVP CPCM(Water), TightSCF, RIJCOSX, def2/J.\n")
        f.write("Panel includes the original seven DFT anchors plus three added FM2 anchors.\n\n")
        f.write(merged.to_string(index=False))
        f.write("\n\nOutput CSV:\n")
        f.write(str(OUT_CSV) + "\n")

    print("\n10-compound DFT panel summary generated.")
    print(f"CSV   : {OUT_CSV}")
    print(f"Report: {OUT_REPORT}")

    key_cols = [
        "Compound_ID", "FM", "N_count", "delta_Consensus", "DFT_role",
        "optfreq_terminated_normally", "sp_terminated_normally",
        "n_imaginary_lt_minus20", "HOMO_eV", "LUMO_eV", "Gap_eV", "dipole_D",
        "charge_all_exists"
    ]
    key_cols = [c for c in key_cols if c in merged.columns]

    print("\nKey panel summary:")
    print(merged[key_cols].to_string(index=False))

    incomplete = merged[
        (~merged["optfreq_terminated_normally"].fillna(False)) |
        (~merged["sp_terminated_normally"].fillna(False)) |
        (merged["n_imaginary_lt_minus20"].fillna(999) > 0)
    ]

    if len(incomplete):
        print("\nWARNING: Incomplete/problematic DFT entries:")
        print(incomplete[["Compound_ID", "optfreq_terminated_normally", "sp_terminated_normally", "n_imaginary_lt_minus20"]].to_string(index=False))
    else:
        print("\nAll detected DFT entries have normal Opt/Freq + SP termination and no imaginary modes < -20 cm^-1.")


if __name__ == "__main__":
    main()