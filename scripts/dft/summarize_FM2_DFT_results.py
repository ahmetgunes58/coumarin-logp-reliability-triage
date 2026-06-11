# -*- coding: utf-8 -*-
"""
Summarise completed FM2 DFT anchor calculations.

Targets:
- CMR_GOLD_090
- CMR_GOLD_020
- CMR_GOLD_092

Outputs:
- Dataset_S32_FM2_DFT_anchor_results.csv
- Dataset_S32_FM2_DFT_anchor_status_report.txt
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DATA = ROOT / "data" / "processed"
DFT = ROOT / "dft" / "molecules"

TARGET_IDS = ["CMR_GOLD_090", "CMR_GOLD_020", "CMR_GOLD_092"]

DATASET = DATA / "Dataset_S1_benchmark_dataset.csv"

OUT_CSV = DATA / "Dataset_S32_FM2_DFT_anchor_results.csv"
OUT_REPORT = DATA / "Dataset_S32_FM2_DFT_anchor_status_report.txt"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def parse_final_energy(text: str):
    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    return float(matches[-1]) if matches else np.nan


def parse_termination(text: str):
    return "ORCA TERMINATED NORMALLY" in text


def parse_frequencies(text: str):
    freqs = []
    pat = re.compile(r"^\s*\d+:\s*(-?\d+\.\d+)\s*cm\*\*-1")
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            freqs.append(float(m.group(1)))
    freqs = sorted(freqs)
    neg_lt_20 = [f for f in freqs if f < -20.0]
    small_neg = [f for f in freqs if -20.0 <= f < 0.0]
    nonzero_pos = [f for f in freqs if f > 0.0]
    return {
        "n_frequencies": len(freqs),
        "n_imaginary_lt_minus20": len(neg_lt_20),
        "imaginary_lt_minus20": ";".join(f"{x:.2f}" for x in neg_lt_20),
        "small_negative_modes": ";".join(f"{x:.2f}" for x in small_neg),
        "lowest_positive_frequency_cm-1": nonzero_pos[0] if nonzero_pos else np.nan,
        "lowest_10_frequencies": ";".join(f"{x:.2f}" for x in freqs[:10]),
    }


def parse_orbitals(text: str):
    lines = text.splitlines()
    start_indices = [i for i, line in enumerate(lines) if "ORBITAL ENERGIES" in line]
    if not start_indices:
        return {
            "HOMO_orbital": np.nan,
            "HOMO_eV": np.nan,
            "LUMO_orbital": np.nan,
            "LUMO_eV": np.nan,
            "Gap_eV": np.nan,
        }

    start = start_indices[-1]
    block = []

    for line in lines[start + 1:]:
        upper = line.upper()
        if "MOLECULAR ORBITALS" in upper:
            break
        if "MULLIKEN" in upper:
            break
        if "LOEWDIN" in upper:
            break
        if "DIPOLE MOMENT" in upper:
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


def parse_dipole_debye(text: str):
    matches = re.findall(r"Magnitude\s+\(Debye\)\s*:\s*([+-]?\d+\.\d+)", text)
    if matches:
        return float(matches[-1])
    return np.nan


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {DATASET}")

    df = pd.read_csv(DATASET)
    meta_cols = [
        "Compound_ID", "SMILES", "FM", "N_count", "logP_exp", "Consensus",
        "delta_Consensus", "TPSA", "MW", "Coumarin_Type"
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[df["Compound_ID"].isin(TARGET_IDS)][meta_cols].copy()

    rows = []

    for cid in TARGET_IDS:
        mol_dir = DFT / cid
        opt_out = mol_dir / "output" / f"{cid}_optfreq.out"
        sp_out = mol_dir / "output" / f"{cid}_sp.out"
        opt_xyz = mol_dir / "geometry" / f"{cid}_opt.xyz"

        opt_text = read_text(opt_out)
        sp_text = read_text(sp_out)

        freq_info = parse_frequencies(opt_text)
        orb_info = parse_orbitals(sp_text)

        row = {
            "Compound_ID": cid,
            "optfreq_out": str(opt_out),
            "sp_out": str(sp_out),
            "opt_xyz": str(opt_xyz),
            "optfreq_terminated_normally": parse_termination(opt_text),
            "sp_terminated_normally": parse_termination(sp_text),
            "optfreq_final_energy_Eh": parse_final_energy(opt_text),
            "sp_final_energy_Eh": parse_final_energy(sp_text),
            "dipole_D": parse_dipole_debye(sp_text),
            "opt_xyz_exists": opt_xyz.exists(),
        }
        row.update(freq_info)
        row.update(orb_info)

        rows.append(row)

    res = pd.DataFrame(rows)

    merged = meta.merge(res, on="Compound_ID", how="right")

    # Clean numerical precision
    for c in ["HOMO_eV", "LUMO_eV", "Gap_eV", "dipole_D"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").round(4)

    for c in ["optfreq_final_energy_Eh", "sp_final_energy_Eh"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").round(12)

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("FM2 DFT anchor status report\n")
        f.write("=" * 70 + "\n\n")
        f.write("Protocol: B3LYP-D3BJ/def2-SVP CPCM(Water), TightSCF, RIJCOSX, def2/J\n")
        f.write("Opt/Freq and SP protocol matched to the existing DFT panel.\n\n")
        f.write(merged.to_string(index=False))
        f.write("\n\nOutput CSV:\n")
        f.write(str(OUT_CSV) + "\n")

    print("\nFM2 DFT summary generated.")
    print(f"CSV   : {OUT_CSV}")
    print(f"Report: {OUT_REPORT}")
    print("\nKey results:")
    key_cols = [
        "Compound_ID", "FM", "N_count", "delta_Consensus",
        "optfreq_terminated_normally", "sp_terminated_normally",
        "n_imaginary_lt_minus20", "HOMO_eV", "LUMO_eV", "Gap_eV", "dipole_D"
    ]
    key_cols = [c for c in key_cols if c in merged.columns]
    print(merged[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()