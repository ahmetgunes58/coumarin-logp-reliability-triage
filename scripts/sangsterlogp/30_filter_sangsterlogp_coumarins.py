# -*- coding: utf-8 -*-
"""
Filter coumarin / coumarin-like compounds from SangsterLogP and check overlap
against the current 95-compound benchmark dataset.

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python scripts\30_filter_sangsterlogp_coumarins.py
"""

from pathlib import Path
import re
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError as exc:
    raise ImportError(
        "RDKit is required for this script.\n"
        "Install in conda, for example:\n"
        "conda install -c conda-forge rdkit\n"
    ) from exc


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

SANGSTER_XLSX = Path(r"C:\Users\Ahmet Gunes\Downloads\Datasets.xlsx")
SANGSTER_SHEET = "SangsterLogP (N=23,520)"

BENCHMARK_CANDIDATES = [
    PROJECT_DIR / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv",
    PROJECT_DIR / "data" / "processed" / "benchmark_dataset.csv",
    PROJECT_DIR / "Dataset_S1_benchmark_dataset.csv",
]

OUT_DIR = PROJECT_DIR / "data" / "external" / "SangsterLogP"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL_CANON = OUT_DIR / "SangsterLogP_main_canonicalized.csv"
OUT_CANDIDATES = OUT_DIR / "SangsterLogP_coumarin_candidates_all.csv"
OUT_STRICT = OUT_DIR / "SangsterLogP_strict_coumarin_candidates.csv"
OUT_NONOVERLAP = OUT_DIR / "SangsterLogP_strict_coumarin_nonoverlap_candidates.csv"
OUT_SWISSADME_SMI = OUT_DIR / "SangsterLogP_strict_coumarin_nonoverlap_for_SwissADME.smi"
OUT_REPORT = OUT_DIR / "SangsterLogP_coumarin_filtering_report.txt"


# ============================================================
# 2. Helpers
# ============================================================

def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_column(df: pd.DataFrame, candidates: list[str], required=True):
    norm_map = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def find_benchmark_file() -> Path | None:
    for p in BENCHMARK_CANDIDATES:
        if p.exists():
            return p
    return None


def mol_from_smiles_safe(smiles):
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles):
    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return None, None, None
    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        can = None
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = None
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = None
    return can, inchikey, formula


def inchikey_first_block(inchikey):
    if not isinstance(inchikey, str) or "-" not in inchikey:
        return None
    return inchikey.split("-")[0]


# ============================================================
# 3. Coumarin SMARTS definitions
# ============================================================
# Strict core: 2H-chromen-2-one / coumarin core.
# Additional 4-hydroxycoumarin-like pattern is included because some
# biologically relevant coumarins are represented in tautomeric/diketo forms.

SMARTS_PATTERNS = {
    "strict_coumarin_core": "O=C1Oc2ccccc2C=C1",
    "aromatic_coumarin_core": "O=c1oc2ccccc2cc1",
    "four_hydroxycoumarin_like": "O=C1OC2=CC=CC=C2C(=O)C1",
}

compiled_smarts = {}
for name, smarts in SMARTS_PATTERNS.items():
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"Invalid SMARTS pattern: {name} -> {smarts}")
    compiled_smarts[name] = patt


def match_coumarin_patterns(smiles):
    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return {
            "valid_mol": False,
            "strict_coumarin_core": False,
            "aromatic_coumarin_core": False,
            "four_hydroxycoumarin_like": False,
            "strict_coumarin_any": False,
        }

    results = {"valid_mol": True}
    for name, patt in compiled_smarts.items():
        results[name] = bool(mol.HasSubstructMatch(patt))

    results["strict_coumarin_any"] = any(
        results[k] for k in [
            "strict_coumarin_core",
            "aromatic_coumarin_core",
            "four_hydroxycoumarin_like",
        ]
    )
    return results


def name_contains_coumarin(name):
    if pd.isna(name):
        return False
    text = str(name).lower()
    keywords = [
        "coumarin",
        "chromen-2-one",
        "chromenone",
        "benzopyran-2-one",
        "umbelliferone",
        "esculetin",
        "scopoletin",
        "warfarin",
        "dicoumarol",
        "psoralen",
        "angelicin",
    ]
    return any(k in text for k in keywords)


# ============================================================
# 4. Load SangsterLogP main sheet
# ============================================================

if not SANGSTER_XLSX.exists():
    raise FileNotFoundError(f"SangsterLogP Excel file not found:\n{SANGSTER_XLSX}")

sang = pd.read_excel(SANGSTER_XLSX, sheet_name=SANGSTER_SHEET)

id_col = find_column(sang, ["ID", "compound_id", "identifier"], required=False)
name_col = find_column(sang, ["Name", "compound", "compound_name"], required=False)
cas_col = find_column(sang, ["CASRN", "CAS", "cas_number"], required=False)
smiles_col = find_column(sang, ["SMILES", "canonical_smiles", "isomeric_smiles"])
logp_col = find_column(sang, ["logP", "experimental_logP", "Exp_logP", "logP_exp"])

keep_cols = []
for col in [id_col, name_col, cas_col, smiles_col, logp_col]:
    if col is not None and col not in keep_cols:
        keep_cols.append(col)

work = sang[keep_cols].copy()
rename_map = {
    smiles_col: "SMILES",
    logp_col: "logP_exp",
}
if id_col:
    rename_map[id_col] = "Sangster_ID"
if name_col:
    rename_map[name_col] = "Name"
if cas_col:
    rename_map[cas_col] = "CASRN"

work = work.rename(columns=rename_map)

if "Sangster_ID" not in work.columns:
    work["Sangster_ID"] = [f"SANGSTER_{i+1:05d}" for i in range(len(work))]
if "Name" not in work.columns:
    work["Name"] = ""
if "CASRN" not in work.columns:
    work["CASRN"] = ""

work["logP_exp"] = pd.to_numeric(work["logP_exp"], errors="coerce")
work = work.dropna(subset=["SMILES", "logP_exp"]).copy()

# Canonicalization
canonical_rows = []
for smiles in work["SMILES"]:
    can, inchikey, formula = canonicalize_smiles(smiles)
    canonical_rows.append((can, inchikey, formula))

work[["Canonical_SMILES", "InChIKey", "Formula"]] = pd.DataFrame(
    canonical_rows,
    index=work.index,
)

work["InChIKey_first_block"] = work["InChIKey"].apply(inchikey_first_block)
work["valid_mol"] = work["Canonical_SMILES"].notna()

# Coumarin pattern matching
pattern_df = pd.DataFrame(
    [match_coumarin_patterns(smi) for smi in work["SMILES"]],
    index=work.index,
)

work = pd.concat([work, pattern_df.drop(columns=["valid_mol"], errors="ignore")], axis=1)
work["name_contains_coumarin"] = work["Name"].apply(name_contains_coumarin)

work["coumarin_candidate_any"] = (
    work["strict_coumarin_any"] | work["name_contains_coumarin"]
)

work.to_csv(OUT_ALL_CANON, index=False, encoding="utf-8-sig")


# ============================================================
# 5. Benchmark overlap check
# ============================================================

benchmark_file = find_benchmark_file()

benchmark_inchikeys = set()
benchmark_first_blocks = set()
benchmark_info = "No benchmark dataset found; overlap check not performed."

if benchmark_file is not None:
    bench = pd.read_csv(benchmark_file)

    bench_smiles_col = find_column(
        bench,
        ["SMILES", "Canonical_SMILES", "canonical_smiles", "smiles"],
        required=False,
    )

    bench_inchikey_col = find_column(
        bench,
        ["InChIKey", "inchikey", "InChI_Key"],
        required=False,
    )

    if bench_inchikey_col is not None:
        keys = bench[bench_inchikey_col].dropna().astype(str).str.strip()
        benchmark_inchikeys.update(keys)
        benchmark_first_blocks.update(
            k.split("-")[0] for k in keys if "-" in k
        )

    elif bench_smiles_col is not None:
        bench_keys = []
        for smi in bench[bench_smiles_col]:
            _, key, _ = canonicalize_smiles(smi)
            if key:
                bench_keys.append(key)

        benchmark_inchikeys.update(bench_keys)
        benchmark_first_blocks.update(
            k.split("-")[0] for k in bench_keys if "-" in k
        )

    else:
        benchmark_info = (
            f"Benchmark dataset found at {benchmark_file}, but no SMILES/InChIKey column detected."
        )

    if benchmark_inchikeys:
        benchmark_info = (
            f"Benchmark dataset: {benchmark_file}\n"
            f"Benchmark exact InChIKeys: {len(benchmark_inchikeys)}\n"
            f"Benchmark first-block InChIKeys: {len(benchmark_first_blocks)}"
        )

work["overlap_exact_inchikey_with_95"] = work["InChIKey"].isin(benchmark_inchikeys)
work["overlap_first_block_with_95"] = work["InChIKey_first_block"].isin(benchmark_first_blocks)


# ============================================================
# 6. Candidate sets
# ============================================================

candidates_any = work[work["coumarin_candidate_any"]].copy()

strict_candidates = work[
    work["strict_coumarin_any"]
].copy()

strict_nonoverlap = strict_candidates[
    ~strict_candidates["overlap_exact_inchikey_with_95"]
].copy()

# Strongest independent set: exact non-overlap and first-block non-overlap
strict_nonoverlap_strong = strict_candidates[
    (~strict_candidates["overlap_exact_inchikey_with_95"])
    & (~strict_candidates["overlap_first_block_with_95"])
].copy()

# Save all strict non-overlap by exact key; include first-block flag for manual review
candidates_any.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8-sig")
strict_candidates.to_csv(OUT_STRICT, index=False, encoding="utf-8-sig")
strict_nonoverlap.to_csv(OUT_NONOVERLAP, index=False, encoding="utf-8-sig")

# SwissADME input file for strongest independent candidates
with open(OUT_SWISSADME_SMI, "w", encoding="utf-8") as f:
    for _, row in strict_nonoverlap_strong.iterrows():
        compound_id = f"SANG_CMR_{int(row.name)+1:05d}"
        smi = row["Canonical_SMILES"]
        if pd.notna(smi):
            f.write(f"{smi} {compound_id}\n")


# ============================================================
# 7. Report
# ============================================================

report = []

report.append("SangsterLogP coumarin filtering report")
report.append("=" * 72)
report.append(f"Input Excel: {SANGSTER_XLSX}")
report.append(f"Sheet: {SANGSTER_SHEET}")
report.append("")
report.append("Column mapping")
report.append("-" * 72)
report.append(f"ID column:     {id_col}")
report.append(f"Name column:   {name_col}")
report.append(f"CASRN column:  {cas_col}")
report.append(f"SMILES column: {smiles_col}")
report.append(f"logP column:   {logp_col}")
report.append("")
report.append("Benchmark overlap source")
report.append("-" * 72)
report.append(benchmark_info)
report.append("")
report.append("SangsterLogP filtering summary")
report.append("-" * 72)
report.append(f"Rows in selected sheet: {len(sang)}")
report.append(f"Rows with SMILES and numeric logP: {len(work)}")
report.append(f"Valid canonical molecules: {int(work['valid_mol'].sum())}")
report.append("")
report.append("Coumarin pattern counts")
report.append("-" * 72)
for key in [
    "strict_coumarin_core",
    "aromatic_coumarin_core",
    "four_hydroxycoumarin_like",
    "strict_coumarin_any",
    "name_contains_coumarin",
    "coumarin_candidate_any",
]:
    report.append(f"{key}: {int(work[key].sum())}")
report.append("")
report.append("Overlap summary for strict coumarin candidates")
report.append("-" * 72)
report.append(f"Strict coumarin candidates: {len(strict_candidates)}")
report.append(f"Exact InChIKey overlap with current 95: {int(strict_candidates['overlap_exact_inchikey_with_95'].sum())}")
report.append(f"First-block InChIKey overlap with current 95: {int(strict_candidates['overlap_first_block_with_95'].sum())}")
report.append(f"Strict coumarin exact non-overlap candidates: {len(strict_nonoverlap)}")
report.append(f"Strict coumarin strong non-overlap candidates: {len(strict_nonoverlap_strong)}")
report.append("")
report.append("Output files")
report.append("-" * 72)
report.append(f"All canonicalized Sangster data: {OUT_ALL_CANON}")
report.append(f"All coumarin/name candidates:    {OUT_CANDIDATES}")
report.append(f"Strict coumarin candidates:      {OUT_STRICT}")
report.append(f"Strict non-overlap candidates:   {OUT_NONOVERLAP}")
report.append(f"SwissADME SMI input:             {OUT_SWISSADME_SMI}")
report.append("")
report.append("Interpretation guide")
report.append("-" * 72)
report.append("If strict strong non-overlap candidates >= 20, the set is likely useful as an external audit.")
report.append("If the count is 5-19, use cautiously as an external illustrative/stress-test subset.")
report.append("If the count is <5, do not expand the manuscript around this dataset.")
report.append("")
report.append("Important wording")
report.append("-" * 72)
report.append("Do not call this a prospective validation unless predictions were truly generated before seeing experimental labels.")
report.append("Recommended wording: non-overlapping external coumarin audit or external stress-test subset.")

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\nSangsterLogP coumarin filtering completed.")
print(f"Report: {OUT_REPORT}")
print("")
print("\n".join(report[:70]))