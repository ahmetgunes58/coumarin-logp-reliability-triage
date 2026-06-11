from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DATA = ROOT / "data" / "processed"
OUT = DATA / "Dataset_S28_FM2_DFT_anchor_candidates.csv"

dataset_path = DATA / "Dataset_S1_benchmark_dataset.csv"

if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset bulunamadı: {dataset_path}")

df = pd.read_csv(dataset_path)

required = ["Compound_ID", "FM", "logP_exp", "Consensus", "delta_Consensus", "N_count"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Eksik kolonlar: {missing}\nMevcut kolonlar: {list(df.columns)}")

# Optional columns
optional_cols = [
    "SMILES",
    "Coumarin_Type",
    "Structural_Class",
    "TPSA",
    "MW",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos_IT",
]

cols = required + [c for c in optional_cols if c in df.columns]

fm2 = df[df["FM"].astype(str).str.upper().eq("FM2")].copy()

if fm2.empty:
    raise ValueError("FM2 bileşik bulunamadı.")

# Numeric cleanup
for c in ["logP_exp", "Consensus", "delta_Consensus", "N_count", "TPSA", "MW"]:
    if c in fm2.columns:
        fm2[c] = pd.to_numeric(fm2[c], errors="coerce")

# Spread4 if possible
fragment_cols = [c for c in ["XLOGP3", "WLOGP", "MLOGP", "Silicos_IT"] if c in fm2.columns]
if len(fragment_cols) == 4:
    for c in fragment_cols:
        fm2[c] = pd.to_numeric(fm2[c], errors="coerce")
    fm2["Spread4"] = fm2[fragment_cols].max(axis=1) - fm2[fragment_cols].min(axis=1)
else:
    fm2["Spread4"] = np.nan

fm2["abs_delta_Consensus"] = fm2["delta_Consensus"].abs()

# FM2 class target values
fm2_mean_bias = fm2["delta_Consensus"].mean()
fm2_median_bias = fm2["delta_Consensus"].median()
fm2_mean_abs = fm2["abs_delta_Consensus"].mean()

fm2["distance_to_FM2_mean_bias"] = (fm2["delta_Consensus"] - fm2_mean_bias).abs()
fm2["distance_to_FM2_median_bias"] = (fm2["delta_Consensus"] - fm2_median_bias).abs()

# Heuristic categories
fm2["Candidate_type"] = ""

# Representative: close to FM2 mean bias
rep_cut = fm2["distance_to_FM2_mean_bias"].quantile(0.20)
fm2.loc[fm2["distance_to_FM2_mean_bias"] <= rep_cut, "Candidate_type"] += "Representative_FM2; "

# Severe: absolute error >= 2
fm2.loc[fm2["abs_delta_Consensus"] >= 2.0, "Candidate_type"] += "Severe_FM2; "

# High Spread4: top quartile within FM2
if fm2["Spread4"].notna().any():
    spread_cut = fm2["Spread4"].quantile(0.75)
    fm2.loc[fm2["Spread4"] >= spread_cut, "Candidate_type"] += "High_Spread4; "

fm2["Candidate_type"] = fm2["Candidate_type"].str.strip()

# Ranking priority:
# 1 representative candidates with moderate/clear error
# 2 high Spread4
# 3 severe candidates
fm2["priority_score"] = 0
fm2.loc[fm2["Candidate_type"].str.contains("Representative_FM2", na=False), "priority_score"] += 3
fm2.loc[fm2["Candidate_type"].str.contains("High_Spread4", na=False), "priority_score"] += 2
fm2.loc[fm2["Candidate_type"].str.contains("Severe_FM2", na=False), "priority_score"] += 1

sort_cols = ["priority_score", "distance_to_FM2_mean_bias", "Spread4", "abs_delta_Consensus"]
fm2 = fm2.sort_values(
    sort_cols,
    ascending=[False, True, False, False]
)

output_cols = [
    "Compound_ID",
    "SMILES" if "SMILES" in fm2.columns else None,
    "FM",
    "N_count",
    "logP_exp",
    "Consensus",
    "delta_Consensus",
    "abs_delta_Consensus",
    "Spread4",
    "TPSA" if "TPSA" in fm2.columns else None,
    "MW" if "MW" in fm2.columns else None,
    "Coumarin_Type" if "Coumarin_Type" in fm2.columns else None,
    "Structural_Class" if "Structural_Class" in fm2.columns else None,
    "distance_to_FM2_mean_bias",
    "Candidate_type",
    "priority_score",
]
output_cols = [c for c in output_cols if c is not None and c in fm2.columns]

fm2[output_cols].to_csv(OUT, index=False, encoding="utf-8-sig")

print("\nFM2 DFT anchor candidate extraction tamamlandı.")
print(f"Output: {OUT}")
print(f"FM2 n = {len(fm2)}")
print(f"FM2 mean bias = {fm2_mean_bias:.3f}")
print(f"FM2 median bias = {fm2_median_bias:.3f}")
print(f"FM2 mean abs error = {fm2_mean_abs:.3f}")

print("\nTop 15 FM2 anchor candidates:")
print(fm2[output_cols].head(15).to_string(index=False))