from pathlib import Path
import shutil
import pandas as pd

root = Path.cwd()

paths = {
    "S0": root / "data/raw/Dataset_S0_raw_literature_collection.xlsx",
    "S1": root / "data/processed/Dataset_S1_benchmark_dataset.csv",
    "S2": root / "data/processed/Dataset_S2_experimental_sources.csv",
}

# Backup first
for label, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    backup = path.with_suffix(path.suffix + ".bak_before_STRICT_GOLD")
    shutil.copy2(path, backup)
    print(f"BACKUP: {backup}")

# Update S0 Excel
s0 = pd.read_excel(paths["S0"])
s0["Data_Tier"] = s0["Data_Tier"].replace({"STRICT": "STRICT-GOLD"})
s0.to_excel(paths["S0"], index=False)

# Update S1 CSV
s1 = pd.read_csv(paths["S1"])
s1["Data_Tier"] = s1["Data_Tier"].replace({"STRICT": "STRICT-GOLD"})
s1.to_csv(paths["S1"], index=False, encoding="utf-8-sig")

# Update S2 CSV
s2 = pd.read_csv(paths["S2"])
s2["Data_Tier"] = s2["Data_Tier"].replace({"STRICT": "STRICT-GOLD"})
s2.to_csv(paths["S2"], index=False, encoding="utf-8-sig")

# Validation: IDs and median logP still match
s0_check = pd.read_excel(paths["S0"])
s1_check = pd.read_csv(paths["S1"])
s2_check = pd.read_csv(paths["S2"])

raw_ids = set(s0_check["Compound_ID"])
s1_ids = set(s1_check["Compound_ID"])
s2_ids = set(s2_check["Compound_ID"])

if raw_ids != s1_ids:
    raise ValueError("S0 and S1 Compound_ID sets do not match after relabeling.")
if s1_ids != s2_ids:
    raise ValueError("S1 and S2 Compound_ID sets do not match after relabeling.")

med = (
    s0_check.groupby("Compound_ID")["Experimental_logP"]
    .median()
    .rename("median_raw_logP")
    .reset_index()
)

chk = s1_check[["Compound_ID", "logP_exp"]].merge(med, on="Compound_ID", how="left")
chk["abs_diff"] = (chk["logP_exp"] - chk["median_raw_logP"]).abs()

mismatch_count = int((chk["abs_diff"] > 1e-6).sum())
if mismatch_count != 0:
    raise ValueError(f"Median logP mismatch after relabeling: {mismatch_count}")

print("\nS0 Data_Tier counts:")
print(s0_check["Data_Tier"].value_counts().to_string())

print("\nS1 Data_Tier counts:")
print(s1_check["Data_Tier"].value_counts().to_string())

print("\nS2 Data_Tier counts:")
print(s2_check["Data_Tier"].value_counts().to_string())

print("\nValidation passed: tier relabeling changed labels only; Compound_ID sets and median logP values remain unchanged.")