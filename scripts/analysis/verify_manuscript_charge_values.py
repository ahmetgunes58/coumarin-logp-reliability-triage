# -*- coding: utf-8 -*-
"""
Verify final Mulliken and Loewdin nitrogen charges for manuscript Section 2.10.

Reads:
    data/processed/Dataset_S14_DFT_Mulliken_charges.csv
    data/processed/Dataset_S15_DFT_Lowdin_charges.csv
    data/processed/Dataset_S14b_DFT_N_charge_summary.csv

Target compounds:
    CMR_GOLD_043
    CMR_GOLD_058
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MULLIKEN_FILE = DATA_DIR / "Dataset_S14_DFT_Mulliken_charges.csv"
LOWDIN_FILE = DATA_DIR / "Dataset_S15_DFT_Lowdin_charges.csv"
SUMMARY_FILE = DATA_DIR / "Dataset_S14b_DFT_N_charge_summary.csv"

TARGETS = ["CMR_GOLD_043", "CMR_GOLD_058"]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep original column names but create predictable aliases if needed.
    rename_map = {}

    for col in df.columns:
        low = col.lower().strip()

        if low in {"compound", "compound_id"}:
            rename_map[col] = "Compound"
        elif low in {"atom_index_orca", "orca atom index", "atom_index", "atom index"}:
            rename_map[col] = "Atom_index_ORCA"
        elif low == "element":
            rename_map[col] = "Element"
        elif low in {"charge", "mulliken charge", "lowdin charge", "löwdin charge"}:
            rename_map[col] = "Charge"

    return df.rename(columns=rename_map)


def nitrogen_rows(df: pd.DataFrame, charge_label: str) -> pd.DataFrame:
    df = standardise_columns(df)

    required = {"Compound", "Atom_index_ORCA", "Element", "Charge"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{charge_label}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[
        (df["Compound"].isin(TARGETS)) &
        (df["Element"].astype(str).str.upper() == "N")
    ].copy()

    out["Charge_type"] = charge_label
    out = out[["Compound", "Atom_index_ORCA", "Element", "Charge_type", "Charge"]]
    out = out.sort_values(["Compound", "Atom_index_ORCA", "Charge_type"])
    return out


def main() -> None:
    print("=" * 80)
    print("Verifying final ORCA nitrogen charges for manuscript Section 2.10")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    mulliken = load_csv(MULLIKEN_FILE)
    lowdin = load_csv(LOWDIN_FILE)

    mulliken_n = nitrogen_rows(mulliken, "Mulliken")
    lowdin_n = nitrogen_rows(lowdin, "Lowdin")

    merged = pd.merge(
        mulliken_n.rename(columns={"Charge": "Mulliken_charge"}).drop(columns=["Charge_type"]),
        lowdin_n.rename(columns={"Charge": "Lowdin_charge"}).drop(columns=["Charge_type"]),
        on=["Compound", "Atom_index_ORCA", "Element"],
        how="outer",
    )

    merged = merged.sort_values(["Compound", "Atom_index_ORCA"])

    print("Nitrogen charge values extracted from Dataset_S14 and Dataset_S15:")
    print(merged.to_string(index=False))
    print()

    if SUMMARY_FILE.exists():
        print("-" * 80)
        print("Nitrogen-charge summary file exists:")
        print(SUMMARY_FILE)
        summary = pd.read_csv(SUMMARY_FILE)
        print()
        print("Preview of Dataset_S14b_DFT_N_charge_summary.csv:")
        print(summary.to_string(index=False))
    else:
        print("-" * 80)
        print("Summary file not found:")
        print(SUMMARY_FILE)

    print()
    print("=" * 80)
    print("Manuscript-ready values:")
    print("=" * 80)

    for compound in TARGETS:
        sub = merged[merged["Compound"] == compound]
        print(f"\n{compound}")
        if sub.empty:
            print("  No nitrogen rows found.")
            continue

        mulliken_values = ", ".join(f"{v:.6f}" for v in sub["Mulliken_charge"])
        lowdin_values = ", ".join(f"{v:.6f}" for v in sub["Lowdin_charge"])
        atom_indices = ", ".join(str(int(v)) for v in sub["Atom_index_ORCA"])

        print(f"  ORCA N atom indices : {atom_indices}")
        print(f"  Mulliken charges   : {mulliken_values}")
        print(f"  Lowdin charges     : {lowdin_values}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()