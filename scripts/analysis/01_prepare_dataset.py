# =============================================================================
# 01_prepare_dataset.py
# coumarin-logp Benchmark Pipeline — Step 1: Dataset Preparation
#
# Author  : Ahmet GÜNEŞ
# Affil.  : National Defence University, Turkish Naval Academy,
#           Department of Basic Sciences, Istanbul, Türkiye
# Contact : ahmet.gunes3@msu.edu.tr
# Version : 2.0 (RDKit-based, fully reproducible)
# Date    : 2026
#
# Description:
#   Merges raw experimental logP data with SwissADME predictions,
#   computes all derived features using RDKit, and produces the
#   canonical benchmark_dataset.csv used by all downstream scripts.
#
# Inputs:
#   data/raw/raw_dataset_original.xlsx
#   data/raw/swissadme.csv
#
# Outputs:
#   data/processed/benchmark_dataset.csv   — main analysis dataset
#   data/processed/dataset_summary.txt     — QC report
#
# Usage:
#   cd <project_root>
#   python scripts/01_prepare_dataset.py
#
# Requirements:
#   rdkit >= 2022.09  |  pandas >= 1.5  |  numpy >= 1.23
#
# Reproducibility:
#   All derived features are computed deterministically from SMILES.
#   No random operations. Output is fully reproducible across platforms.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# RDKit — hard dependency
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")          # suppress RDKit warnings
except ImportError:
    sys.exit(
        "\n[ERROR] RDKit not found.\n"
        "Install: conda install -c conda-forge rdkit\n"
    )

# ---------------------------------------------------------------------------
# CONFIGURATION — edit paths here if needed
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")

RAW_EXCEL  = os.path.join(RAW_DIR,  "raw_dataset_original.xlsx")
SWISSADME  = os.path.join(RAW_DIR,  "swissadme.csv")
OUT_CSV    = os.path.join(PROC_DIR, "benchmark_dataset.csv")
OUT_REPORT = os.path.join(PROC_DIR, "dataset_summary.txt")

LIT_RMSE   = 0.60          # Mannhold et al. J. Pharm. Sci. 98 (2009) 861-893

# SwissADME column → internal name mapping
PREDICTOR_MAP = {
    "iLOGP"         : "iLOGP",
    "XLOGP3"        : "XLOGP3",
    "WLOGP"         : "WLOGP",
    "MLOGP"         : "MLOGP",
    "Silicos_IT"    : "Silicos-IT Log P",
    "Consensus"     : "Consensus Log P",
}

# logP range boundaries (for logP_range column, used in Table 4)
LOGP_BINS   = [-np.inf, 1.0, 2.0, 3.0, np.inf]
LOGP_LABELS = ["logP<1", "logP 1-2", "logP 2-3", "logP>3"]

# N group boundaries
N_BINS      = [-1, 0, 1, 3, 100]
N_LABELS    = ["N=0", "N=1", "N=2-3", "N≥4"]

# Failure mode assignment (used in Table 6 / Figure 4)
def assign_fm(n_count: int, logp_exp: float) -> str:
    """
    Five-class failure mode taxonomy.
    FM0: accurate (N-free or compact isolated N)
    FM1: polar overestimation (N=1-3, logP < 1.5)
    FM2: multi-N misassignment (N=1-3, logP >= 1.5)
    FM3: N-driven cancellation (N >= 4)
    FM4: conjugation overflow (N=0, logP > 3)
    """
    if n_count == 0 and logp_exp <= 3.0:
        return "FM0"
    elif 1 <= n_count <= 3 and logp_exp < 1.5:
        return "FM1"
    elif 1 <= n_count <= 3 and logp_exp >= 1.5:
        return "FM2"
    elif n_count >= 4:
        return "FM3"
    elif n_count == 0 and logp_exp > 3.0:
        return "FM4"
    return "FM0"   # fallback


# ---------------------------------------------------------------------------
# RDKit descriptor functions
# ---------------------------------------------------------------------------

def mol_from_smiles(smi: str):
    """Parse SMILES; return None if invalid."""
    mol = Chem.MolFromSmiles(str(smi).strip())
    return mol


def count_nitrogen(smi: str) -> int:
    """
    Count all nitrogen atoms using RDKit.
    Captures both aromatic (n) and aliphatic (N) nitrogens —
    unlike SMILES string counting which misses aromatic n.
    Returns -1 if SMILES is invalid.
    """
    mol = mol_from_smiles(smi)
    if mol is None:
        return -1
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)


def calc_tpsa(smi: str) -> float:
    """Calculate TPSA using RDKit (Å²). Returns NaN if invalid."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return np.nan
    return rdMolDescriptors.CalcTPSA(mol)


def calc_mw(smi: str) -> float:
    """Calculate exact molecular weight. Returns NaN if invalid."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return np.nan
    return Descriptors.ExactMolWt(mol)


def calc_hba(smi: str) -> int:
    """H-bond acceptor count (Lipinski definition)."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return -1
    return rdMolDescriptors.CalcNumHBA(mol)


def calc_hbd(smi: str) -> int:
    """H-bond donor count (Lipinski definition)."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return -1
    return rdMolDescriptors.CalcNumHBD(mol)


def calc_rotbonds(smi: str) -> int:
    """Rotatable bond count."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return -1
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def validate_smiles(smi: str) -> bool:
    """Return True if SMILES is valid."""
    return mol_from_smiles(smi) is not None


# ---------------------------------------------------------------------------
# STEP 1 — Load raw experimental data
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    print(f"\n[1/6] Loading raw dataset: {path}")
    raw = pd.read_excel(path)
    print(f"      Raw rows   : {len(raw)}")
    print(f"      Columns    : {list(raw.columns)}")
    print(f"      Unique IDs : {raw['Compound_ID'].nunique()}")

    # Aggregate replicates — use median logP per compound
    df = raw.groupby("Compound_ID", as_index=False).agg(
        logP_exp      = ("Experimental_logP", "median"),
        n_replicates  = ("Experimental_logP", "count"),
        SMILES        = ("Canonical_SMILES",   "first"),
        Data_Tier     = ("Data_Tier",          "first"),
        Coumarin_Type = ("Coumarin_Type",      "first"),
        # Keep raw TPSA/MW from dataset for cross-check (will be recalculated)
        TPSA_raw      = ("TPSA",               "first"),
        MW_raw        = ("Molecular_Weight",   "first"),
    )

    print(f"      After aggregation: {len(df)} unique compounds")
    print(f"      Data tiers: {df['Data_Tier'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — Load SwissADME predictions
# ---------------------------------------------------------------------------

def load_swissadme(path: str) -> pd.DataFrame:
    print(f"\n[2/6] Loading SwissADME predictions: {path}")
    sw = pd.read_csv(path)
    print(f"      Rows    : {len(sw)}")
    print(f"      Columns : {list(sw.columns)}")

    # Extract only predictor columns and rename
    pred_cols = list(PREDICTOR_MAP.values())
    missing = [c for c in pred_cols if c not in sw.columns]
    if missing:
        sys.exit(f"[ERROR] SwissADME columns not found: {missing}")

    sw_pred = sw[pred_cols].copy()
    sw_pred.columns = list(PREDICTOR_MAP.keys())
    return sw_pred


# ---------------------------------------------------------------------------
# STEP 3 — Merge
# ---------------------------------------------------------------------------

def merge(df: pd.DataFrame, sw_pred: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[3/6] Merging datasets")
    if len(df) != len(sw_pred):
        sys.exit(
            f"[ERROR] Row count mismatch: "
            f"experimental={len(df)}, SwissADME={len(sw_pred)}\n"
            f"Ensure both files are sorted in the same compound order."
        )
    df = df.reset_index(drop=True)
    sw_pred = sw_pred.reset_index(drop=True)
    df = pd.concat([df, sw_pred], axis=1)
    print(f"      Merged: {len(df)} compounds × {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# STEP 4 — RDKit descriptors
# ---------------------------------------------------------------------------

def add_rdkit_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[4/6] Computing RDKit descriptors")

    # Validate all SMILES first
    df["SMILES_valid"] = df["SMILES"].apply(validate_smiles)
    n_invalid = (~df["SMILES_valid"]).sum()
    if n_invalid > 0:
        print(f"      [WARNING] {n_invalid} invalid SMILES:")
        print(df[~df["SMILES_valid"]][["Compound_ID","SMILES"]].to_string(index=False))

    # Nitrogen count — RDKit (correct: counts aromatic n too)
    print("      Computing N_count (RDKit, all nitrogen atoms)...")
    df["N_count"] = df["SMILES"].apply(count_nitrogen)

    # Cross-check against SMILES uppercase N count
    df["N_count_smiles"] = df["SMILES"].str.count("N")
    n_diff = (df["N_count"] != df["N_count_smiles"]).sum()
    if n_diff > 0:
        print(f"      [INFO] {n_diff} compounds differ between RDKit N_count and SMILES uppercase count")
        print(f"             (expected — aromatic nitrogens written as lowercase 'n' in SMILES)")
        diff_examples = df[df["N_count"] != df["N_count_smiles"]][
            ["Compound_ID","N_count","N_count_smiles","SMILES"]
        ].head(5)
        print(diff_examples.to_string(index=False))
    df.drop(columns=["N_count_smiles"], inplace=True)

    # TPSA — RDKit (recalculated for reproducibility)
    print("      Computing TPSA (RDKit)...")
    df["TPSA"] = df["SMILES"].apply(calc_tpsa)

    # Cross-check with raw TPSA
    tpsa_diff = (df["TPSA"] - df["TPSA_raw"]).abs()
    n_tpsa_diff = (tpsa_diff > 1.0).sum()
    if n_tpsa_diff > 0:
        print(f"      [WARNING] {n_tpsa_diff} compounds have TPSA difference > 1 Å² vs raw dataset")

    # MW — RDKit
    print("      Computing MW (RDKit)...")
    df["MW"] = df["SMILES"].apply(calc_mw)

    # Additional descriptors (for SI and extended analyses)
    print("      Computing HBA, HBD, RotBonds...")
    df["HBA"]      = df["SMILES"].apply(calc_hba)
    df["HBD"]      = df["SMILES"].apply(calc_hbd)
    df["RotBonds"] = df["SMILES"].apply(calc_rotbonds)

    # Clean up raw columns (keep for audit trail in full dataset)
    # df.drop(columns=["TPSA_raw","MW_raw"], inplace=True)

    print(f"      N_count range: {df['N_count'].min()}–{df['N_count'].max()}")
    print(f"      N_count distribution: {df['N_count'].value_counts().sort_index().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# STEP 5 — Derived grouping variables and prediction errors
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[5/6] Adding grouping variables and prediction errors")

    # N group (ordered categorical)
    df["N_group"] = pd.cut(
        df["N_count"],
        bins=N_BINS,
        labels=N_LABELS,
        ordered=True,
    )

    # logP range (ordered categorical, 1.0 boundaries for Table 4)
    df["logP_range"] = pd.cut(
        df["logP_exp"],
        bins=LOGP_BINS,
        labels=LOGP_LABELS,
        ordered=True,
    )

    # logP range with 1.5 boundary (for Table 4c non-additivity analysis)
    df["logP_15"] = pd.cut(
        df["logP_exp"],
        bins=[-np.inf, 1.5, 3.0, np.inf],
        labels=["logP<1.5", "logP 1.5-3.0", "logP>3.0"],
        ordered=True,
    )

    # N group collapsed (for Table 4c: N=1 and N=2-3 → N=1-3)
    df["N_15"] = df["N_group"].map({
        "N=0"  : "N=0",
        "N=1"  : "N=1-3",
        "N=2-3": "N=1-3",
        "N≥4"  : "N≥4",
    })

    # Failure mode
    df["FM"] = df.apply(
        lambda row: assign_fm(row["N_count"], row["logP_exp"]), axis=1
    )

    # Prediction errors: delta = logP_exp - logP_pred
    # Negative delta = predictor overestimates logP
    for pred in PREDICTOR_MAP.keys():
        df[f"delta_{pred}"] = df["logP_exp"] - df[pred]

    print(f"      N_group distribution:")
    for g, n in df["N_group"].value_counts().sort_index().items():
        print(f"        {g}: n={n}")

    print(f"      logP_range distribution:")
    for g, n in df["logP_range"].value_counts().sort_index().items():
        print(f"        {g}: n={n}")

    print(f"      FM distribution:")
    for g, n in df["FM"].value_counts().sort_index().items():
        print(f"        {g}: n={n}")

    return df


# ---------------------------------------------------------------------------
# STEP 6 — Quality control and save
# ---------------------------------------------------------------------------

def qc_and_save(df: pd.DataFrame) -> None:
    print(f"\n[6/6] Quality control and saving")
    os.makedirs(PROC_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 65)
    lines.append("DATASET QC REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    lines.append(f"\nTotal compounds     : {len(df)}")
    lines.append(f"logP range          : {df['logP_exp'].min():.2f} – {df['logP_exp'].max():.2f}")
    lines.append(f"logP mean (SD)      : {df['logP_exp'].mean():.2f} ({df['logP_exp'].std():.2f})")

    lines.append(f"\nData tier distribution:")
    for tier, n in df["Data_Tier"].value_counts().items():
        lines.append(f"  {tier}: {n}")

    lines.append(f"\nN_count distribution (RDKit):")
    for nc, n in df["N_count"].value_counts().sort_index().items():
        lines.append(f"  N={nc}: {n} compounds")

    lines.append(f"\nN_group distribution:")
    for g, n in df["N_group"].value_counts().sort_index().items():
        lines.append(f"  {g}: {n}")

    lines.append(f"\nFailure mode distribution:")
    for fm, n in df["FM"].value_counts().sort_index().items():
        lines.append(f"  {fm}: {n}")

    lines.append(f"\nPredictor summary (Consensus logP):")
    d = df["delta_Consensus"]
    lines.append(f"  Bias   : {d.mean():.3f}")
    lines.append(f"  MAE    : {d.abs().mean():.3f}")
    lines.append(f"  RMSE   : {np.sqrt((d**2).mean()):.3f}")
    lines.append(f"  % overestimated: {(d<0).mean()*100:.1f}%")

    lines.append(f"\nInvalid SMILES      : {(~df['SMILES_valid']).sum()}")
    lines.append(f"Missing N_count     : {(df['N_count']==-1).sum()}")
    lines.append(f"Missing TPSA        : {df['TPSA'].isna().sum()}")

    lines.append(f"\nOutput columns ({len(df.columns)}):")
    lines.append(f"  {list(df.columns)}")

    lines.append("\n" + "=" * 65)
    report = "\n".join(lines)
    print(report)

    # Save report
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✅ QC report  → {OUT_REPORT}")

    # Save dataset — column order
    col_order = [
        "Compound_ID", "SMILES", "logP_exp", "n_replicates",
        "Data_Tier", "Coumarin_Type",
        "N_count", "N_group", "N_15",
        "logP_range", "logP_15", "FM",
        "TPSA", "MW", "HBA", "HBD", "RotBonds",
        "TPSA_raw", "MW_raw", "SMILES_valid",
    ] + list(PREDICTOR_MAP.keys()) + [f"delta_{p}" for p in PREDICTOR_MAP.keys()]

    # Only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df[col_order].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✅ Dataset    → {OUT_CSV}")
    print(f"     {len(df)} compounds × {len(col_order)} columns")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("01_prepare_dataset.py — coumarin-logp Benchmark Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    df      = load_raw(RAW_EXCEL)
    sw_pred = load_swissadme(SWISSADME)
    df      = merge(df, sw_pred)
    df      = add_rdkit_descriptors(df)
    df      = add_derived_columns(df)
    qc_and_save(df)

    print(f"\n✅ Pipeline complete.")
    print(f"   Next step: python scripts/02_statistical_analysis.py")
    print("=" * 65)


if __name__ == "__main__":
    main()