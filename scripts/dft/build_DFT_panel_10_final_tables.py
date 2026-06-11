# -*- coding: utf-8 -*-
"""
Build final 10-compound DFT panel tables after completing Opt/Freq, SP,
frontier orbital, dipole, and Mulliken/Loewdin charge extraction.

Robust version:
- Recomputes heavy-atom flag from element column.
- Handles old/new charge CSV format differences.
- Avoids idxmin/idxmax failure if a malformed empty heavy-atom subset appears.

Outputs:
- Dataset_S37_DFT_panel_10_Mulliken_Loewdin_charges.csv
- Dataset_S37_DFT_panel_10_heavy_atom_charges.csv
- Dataset_S38_DFT_panel_10_charge_summary.csv
- Dataset_S39_DFT_panel_10_frontier_charge_summary.csv
- Dataset_S39_DFT_panel_10_final_report.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np


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

S34 = DATA / "Dataset_S34_DFT_panel_10_summary.csv"

OUT_ALL_CHARGES = DATA / "Dataset_S37_DFT_panel_10_Mulliken_Loewdin_charges.csv"
OUT_HEAVY_CHARGES = DATA / "Dataset_S37_DFT_panel_10_heavy_atom_charges.csv"
OUT_CHARGE_SUMMARY = DATA / "Dataset_S38_DFT_panel_10_charge_summary.csv"
OUT_FINAL_SUMMARY = DATA / "Dataset_S39_DFT_panel_10_frontier_charge_summary.csv"
OUT_REPORT = DATA / "Dataset_S39_DFT_panel_10_final_report.txt"


def load_charge_file(cid):
    p = DFT / cid / "extracted_data" / f"{cid}_charges_mulliken_loewdin.csv"
    if not p.exists():
        raise FileNotFoundError(f"Charge file missing: {p}")

    df = pd.read_csv(p)

    # Normalize common column variants if needed
    rename_map = {}
    if "Element" in df.columns and "element" not in df.columns:
        rename_map["Element"] = "element"
    if "atom" in df.columns and "element" not in df.columns:
        rename_map["atom"] = "element"
    if "Mulliken" in df.columns and "Mulliken_charge" not in df.columns:
        rename_map["Mulliken"] = "Mulliken_charge"
    if "Loewdin" in df.columns and "Loewdin_charge" not in df.columns:
        rename_map["Loewdin"] = "Loewdin_charge"
    if "Lowdin_charge" in df.columns and "Loewdin_charge" not in df.columns:
        rename_map["Lowdin_charge"] = "Loewdin_charge"
    if "Löwdin_charge" in df.columns and "Loewdin_charge" not in df.columns:
        rename_map["Löwdin_charge"] = "Loewdin_charge"

    df = df.rename(columns=rename_map)

    required = ["element", "Mulliken_charge", "Loewdin_charge"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{cid} charge file has missing columns: {missing}. Columns: {list(df.columns)}")

    # Ensure standard atom index columns
    if "atom_index_0based" not in df.columns:
        df["atom_index_0based"] = range(len(df))
    if "atom_index_1based" not in df.columns:
        df["atom_index_1based"] = df["atom_index_0based"] + 1

    df["Compound_ID"] = cid
    df["element"] = df["element"].astype(str).str.strip()

    df["Mulliken_charge"] = pd.to_numeric(df["Mulliken_charge"], errors="coerce")
    df["Loewdin_charge"] = pd.to_numeric(df["Loewdin_charge"], errors="coerce")

    # Recompute heavy atom flag robustly; do not trust old boolean/string format
    df["is_heavy_atom"] = df["element"].str.upper().ne("H")

    return df


def extreme_row(df, charge_col, mode):
    valid = df.dropna(subset=[charge_col]).copy()

    if valid.empty:
        return {
            f"{mode}_{charge_col}_atom_index_1based": np.nan,
            f"{mode}_{charge_col}_element": "",
            f"{mode}_{charge_col}": np.nan,
        }

    if mode == "min":
        idx = valid[charge_col].idxmin()
    elif mode == "max":
        idx = valid[charge_col].idxmax()
    else:
        raise ValueError(mode)

    row = valid.loc[idx]

    return {
        f"{mode}_{charge_col}_atom_index_1based": row.get("atom_index_1based", np.nan),
        f"{mode}_{charge_col}_element": row.get("element", ""),
        f"{mode}_{charge_col}": row.get(charge_col, np.nan),
    }


def main():
    if not S34.exists():
        raise FileNotFoundError(f"S34 summary missing: {S34}")

    panel = pd.read_csv(S34)

    charge_dfs = []
    for cid in PANEL_IDS:
        charge_dfs.append(load_charge_file(cid))

    charges = pd.concat(charge_dfs, ignore_index=True)

    # Final robust heavy atom selection
    charges["is_heavy_atom"] = charges["element"].astype(str).str.upper().ne("H")
    heavy = charges[charges["is_heavy_atom"]].copy()

    charges.to_csv(OUT_ALL_CHARGES, index=False, encoding="utf-8-sig")
    heavy.to_csv(OUT_HEAVY_CHARGES, index=False, encoding="utf-8-sig")

    summary_rows = []

    for cid in PANEL_IDS:
        g = charges[charges["Compound_ID"] == cid].copy()
        gh = g[g["is_heavy_atom"]].copy()

        if gh.empty:
            print(f"WARNING: {cid} has no heavy atoms after parsing. Check charge CSV.")
            gh = g.copy()

        row = {
            "Compound_ID": cid,
            "n_atoms": len(g),
            "n_heavy_atoms": int(g["is_heavy_atom"].sum()),
            "min_Mulliken_charge": gh["Mulliken_charge"].min(),
            "max_Mulliken_charge": gh["Mulliken_charge"].max(),
            "min_Loewdin_charge": gh["Loewdin_charge"].min(),
            "max_Loewdin_charge": gh["Loewdin_charge"].max(),
            "mean_abs_Mulliken_heavy": gh["Mulliken_charge"].abs().mean(),
            "mean_abs_Loewdin_heavy": gh["Loewdin_charge"].abs().mean(),
        }

        row.update(extreme_row(gh, "Mulliken_charge", "min"))
        row.update(extreme_row(gh, "Mulliken_charge", "max"))
        row.update(extreme_row(gh, "Loewdin_charge", "min"))
        row.update(extreme_row(gh, "Loewdin_charge", "max"))

        summary_rows.append(row)

    charge_summary = pd.DataFrame(summary_rows)

    order = {cid: i for i, cid in enumerate(PANEL_IDS)}
    charge_summary["__order"] = charge_summary["Compound_ID"].map(order)
    charge_summary = charge_summary.sort_values("__order").drop(columns="__order")

    for c in charge_summary.columns:
        if charge_summary[c].dtype.kind in "fc":
            charge_summary[c] = charge_summary[c].round(4)

    charge_summary.to_csv(OUT_CHARGE_SUMMARY, index=False, encoding="utf-8-sig")

    selected_panel_cols = [
        "Compound_ID",
        "FM",
        "N_count",
        "logP_exp",
        "Consensus",
        "delta_Consensus",
        "DFT_role",
        "optfreq_terminated_normally",
        "sp_terminated_normally",
        "n_imaginary_lt_minus20",
        "HOMO_eV",
        "LUMO_eV",
        "Gap_eV",
        "dipole_D",
        "charge_all_exists",
    ]
    selected_panel_cols = [c for c in selected_panel_cols if c in panel.columns]

    final_summary = panel[selected_panel_cols].merge(
        charge_summary,
        on="Compound_ID",
        how="left",
    )

    for c in final_summary.columns:
        if final_summary[c].dtype.kind in "fc":
            final_summary[c] = final_summary[c].round(4)

    final_summary.to_csv(OUT_FINAL_SUMMARY, index=False, encoding="utf-8-sig")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("10-compound DFT panel final table report\n")
        f.write("=" * 80 + "\n\n")
        f.write("Generated outputs:\n")
        f.write(str(OUT_ALL_CHARGES) + "\n")
        f.write(str(OUT_HEAVY_CHARGES) + "\n")
        f.write(str(OUT_CHARGE_SUMMARY) + "\n")
        f.write(str(OUT_FINAL_SUMMARY) + "\n\n")
        f.write("Final summary:\n")
        f.write(final_summary.to_string(index=False))
        f.write("\n\n")

    print("\n10-compound DFT final tables generated.")
    print(f"All charges       : {OUT_ALL_CHARGES}")
    print(f"Heavy charges     : {OUT_HEAVY_CHARGES}")
    print(f"Charge summary    : {OUT_CHARGE_SUMMARY}")
    print(f"Final DFT summary : {OUT_FINAL_SUMMARY}")
    print(f"Report            : {OUT_REPORT}")

    print("\nFinal DFT panel summary:")
    show_cols = [
        "Compound_ID",
        "FM",
        "N_count",
        "delta_Consensus",
        "HOMO_eV",
        "LUMO_eV",
        "Gap_eV",
        "dipole_D",
        "min_Mulliken_charge",
        "max_Mulliken_charge",
        "min_Loewdin_charge",
        "max_Loewdin_charge",
    ]
    show_cols = [c for c in show_cols if c in final_summary.columns]
    print(final_summary[show_cols].to_string(index=False))

    missing = final_summary[
        (final_summary["optfreq_terminated_normally"] != True) |
        (final_summary["sp_terminated_normally"] != True) |
        (final_summary["n_imaginary_lt_minus20"] != 0) |
        (final_summary["charge_all_exists"] != True)
    ]

    if len(missing):
        print("\nWARNING: incomplete/problematic entries:")
        print(missing.to_string(index=False))
    else:
        print("\nAll 10 DFT panel entries are complete.")


if __name__ == "__main__":
    main()