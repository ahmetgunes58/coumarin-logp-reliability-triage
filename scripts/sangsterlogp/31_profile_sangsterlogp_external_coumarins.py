# -*- coding: utf-8 -*-
"""
Profile the non-overlapping SangsterLogP strict coumarin candidate set
before running SwissADME / external predictor audit.

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python scripts\31_profile_sangsterlogp_external_coumarins.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError as exc:
    raise ImportError(
        "RDKit is required.\n"
        "Install with: conda install -c conda-forge rdkit"
    ) from exc


PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

IN_FILE = PROJECT_DIR / "data" / "external" / "SangsterLogP" / "SangsterLogP_strict_coumarin_nonoverlap_candidates.csv"

OUT_DIR = PROJECT_DIR / "data" / "external" / "SangsterLogP"
OUT_PROFILE = OUT_DIR / "SangsterLogP_external_coumarin_profile.csv"
OUT_SUMMARY = OUT_DIR / "SangsterLogP_external_coumarin_profile_summary.txt"
OUT_SWISSADME_CSV = OUT_DIR / "SangsterLogP_external_coumarin_for_SwissADME.csv"
OUT_SWISSADME_SMI = OUT_DIR / "SangsterLogP_external_coumarin_for_SwissADME.smi"


def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None


def n_count(mol):
    if mol is None:
        return np.nan
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)


def o_count(mol):
    if mol is None:
        return np.nan
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)


def heavy_atom_count(mol):
    if mol is None:
        return np.nan
    return mol.GetNumHeavyAtoms()


def mw(mol):
    if mol is None:
        return np.nan
    return Descriptors.MolWt(mol)


def tpsa(mol):
    if mol is None:
        return np.nan
    return rdMolDescriptors.CalcTPSA(mol)


def hbd(mol):
    if mol is None:
        return np.nan
    return rdMolDescriptors.CalcNumHBD(mol)


def hba(mol):
    if mol is None:
        return np.nan
    return rdMolDescriptors.CalcNumHBA(mol)


def n_group(n):
    if pd.isna(n):
        return "NA"
    n = int(n)
    if n == 0:
        return "N = 0"
    if n == 1:
        return "N = 1"
    if n in (2, 3):
        return "N = 2–3"
    return "N ≥ 4"


def logp_bin(x):
    if pd.isna(x):
        return "NA"
    x = float(x)
    if x < 1.5:
        return "logP < 1.5"
    if x <= 3.0:
        return "1.5 ≤ logP ≤ 3.0"
    return "logP > 3.0"


def possible_problematic_smiles(smiles):
    """
    Conservative flag for salts/mixtures/dot-disconnected entries.
    Not automatic exclusion; just for manual review.
    """
    if pd.isna(smiles):
        return True
    s = str(smiles)
    return "." in s


if not IN_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{IN_FILE}")

df = pd.read_csv(IN_FILE)

required = ["Canonical_SMILES", "logP_exp", "InChIKey", "Name"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

df["logP_exp"] = pd.to_numeric(df["logP_exp"], errors="coerce")
df = df.dropna(subset=["Canonical_SMILES", "logP_exp"]).copy()

# Strong non-overlap subset: remove first-block overlap as well, if column exists.
if "overlap_first_block_with_95" in df.columns:
    strong = df[df["overlap_first_block_with_95"] == False].copy()
else:
    strong = df.copy()

# Deduplicate exact InChIKey within external set.
before_dedup = len(strong)
strong = strong.drop_duplicates(subset=["InChIKey"]).copy()
after_dedup = len(strong)

mols = [mol_from_smiles(smi) for smi in strong["Canonical_SMILES"]]

strong["valid_mol"] = [m is not None for m in mols]
strong["N_count"] = [n_count(m) for m in mols]
strong["O_count"] = [o_count(m) for m in mols]
strong["Heavy_atoms"] = [heavy_atom_count(m) for m in mols]
strong["MW"] = [mw(m) for m in mols]
strong["TPSA"] = [tpsa(m) for m in mols]
strong["HBD"] = [hbd(m) for m in mols]
strong["HBA"] = [hba(m) for m in mols]
strong["N_group"] = strong["N_count"].apply(n_group)
strong["logP_bin"] = strong["logP_exp"].apply(logp_bin)
strong["dot_disconnected_entry"] = strong["Canonical_SMILES"].apply(possible_problematic_smiles)

# Create stable external IDs
strong = strong.reset_index(drop=True)
strong["External_ID"] = [f"SANG_CMR_EXT_{i+1:03d}" for i in range(len(strong))]

# Reorder columns
preferred_cols = [
    "External_ID",
    "Sangster_ID",
    "Name",
    "CASRN",
    "Canonical_SMILES",
    "InChIKey",
    "logP_exp",
    "N_count",
    "N_group",
    "logP_bin",
    "O_count",
    "MW",
    "TPSA",
    "HBD",
    "HBA",
    "Heavy_atoms",
    "dot_disconnected_entry",
    "strict_coumarin_any",
    "aromatic_coumarin_core",
    "overlap_exact_inchikey_with_95",
    "overlap_first_block_with_95",
]

available_cols = [c for c in preferred_cols if c in strong.columns]
remaining_cols = [c for c in strong.columns if c not in available_cols]
strong = strong[available_cols + remaining_cols]

strong.to_csv(OUT_PROFILE, index=False, encoding="utf-8-sig")

# SwissADME inputs
swiss_cols = ["External_ID", "Canonical_SMILES", "Name", "logP_exp", "N_count", "N_group", "logP_bin"]
strong[swiss_cols].to_csv(OUT_SWISSADME_CSV, index=False, encoding="utf-8-sig")

with open(OUT_SWISSADME_SMI, "w", encoding="utf-8") as f:
    for _, row in strong.iterrows():
        f.write(f"{row['Canonical_SMILES']} {row['External_ID']}\n")

# Summary tables
n_group_counts = strong["N_group"].value_counts().reindex(["N = 0", "N = 1", "N = 2–3", "N ≥ 4"], fill_value=0)
logp_bin_counts = strong["logP_bin"].value_counts().reindex(["logP < 1.5", "1.5 ≤ logP ≤ 3.0", "logP > 3.0"], fill_value=0)
risk_matrix = pd.crosstab(strong["N_group"], strong["logP_bin"]).reindex(
    index=["N = 0", "N = 1", "N = 2–3", "N ≥ 4"],
    columns=["logP < 1.5", "1.5 ≤ logP ≤ 3.0", "logP > 3.0"],
    fill_value=0,
)

summary_lines = []
summary_lines.append("SangsterLogP external strict coumarin profile")
summary_lines.append("=" * 72)
summary_lines.append(f"Input file: {IN_FILE}")
summary_lines.append(f"Exact non-overlap input rows: {len(df)}")
summary_lines.append(f"Strong non-overlap rows before external deduplication: {before_dedup}")
summary_lines.append(f"Strong non-overlap rows after exact InChIKey deduplication: {after_dedup}")
summary_lines.append(f"Valid molecules: {int(strong['valid_mol'].sum())}")
summary_lines.append(f"Dot-disconnected / possible salt-mixture entries: {int(strong['dot_disconnected_entry'].sum())}")
summary_lines.append("")
summary_lines.append("N-count group distribution")
summary_lines.append("-" * 72)
summary_lines.append(n_group_counts.to_string())
summary_lines.append("")
summary_lines.append("Experimental logP-bin distribution")
summary_lines.append("-" * 72)
summary_lines.append(logp_bin_counts.to_string())
summary_lines.append("")
summary_lines.append("N-count × logP-bin matrix")
summary_lines.append("-" * 72)
summary_lines.append(risk_matrix.to_string())
summary_lines.append("")
summary_lines.append("logP descriptive statistics")
summary_lines.append("-" * 72)
summary_lines.append(str(strong["logP_exp"].describe()))
summary_lines.append("")
summary_lines.append("N_count descriptive statistics")
summary_lines.append("-" * 72)
summary_lines.append(str(strong["N_count"].describe()))
summary_lines.append("")
summary_lines.append("Output files")
summary_lines.append("-" * 72)
summary_lines.append(f"Profile CSV:       {OUT_PROFILE}")
summary_lines.append(f"SwissADME CSV:     {OUT_SWISSADME_CSV}")
summary_lines.append(f"SwissADME SMI:     {OUT_SWISSADME_SMI}")

OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

print("\nSangsterLogP external coumarin profiling completed.")
print(f"Profile CSV:   {OUT_PROFILE}")
print(f"Summary TXT:   {OUT_SUMMARY}")
print(f"SwissADME CSV: {OUT_SWISSADME_CSV}")
print(f"SwissADME SMI: {OUT_SWISSADME_SMI}")
print("")
print("\n".join(summary_lines))