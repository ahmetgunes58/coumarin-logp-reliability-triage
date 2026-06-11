import pandas as pd
from pathlib import Path

root = Path.cwd()

raw_path = root / "data/raw/Dataset_S0_raw_literature_collection.xlsx"
s1_path = root / "data/processed/Dataset_S1_benchmark_dataset.csv"

out_s2 = root / "data/processed/Dataset_S2_experimental_sources.csv"
out_s3 = root / "data/processed/Dataset_S3_exclusion_log.csv"

raw = pd.read_excel(raw_path)
s1 = pd.read_csv(s1_path)

raw_ids = set(raw["Compound_ID"])
s1_ids = set(s1["Compound_ID"])

raw_unique = raw["Compound_ID"].nunique()
s1_unique = s1["Compound_ID"].nunique()

med = (
    raw.groupby("Compound_ID")["Experimental_logP"]
    .median()
    .rename("median_raw_logP")
    .reset_index()
)

chk = s1[["Compound_ID", "logP_exp", "n_replicates", "Data_Tier"]].merge(
    med, on="Compound_ID", how="left"
)
chk["abs_diff"] = (chk["logP_exp"] - chk["median_raw_logP"]).abs()
median_mismatch_count = int((chk["abs_diff"] > 1e-6).sum())

if raw_unique != 95 or len(s1) != 95 or s1_unique != 95:
    raise ValueError(
        f"Unexpected dataset size: raw_unique={raw_unique}, "
        f"s1_rows={len(s1)}, s1_unique={s1_unique}"
    )

if raw_ids != s1_ids:
    raise ValueError("Compound_ID mismatch between Dataset_S0 and Dataset_S1.")

if median_mismatch_count != 0:
    raise ValueError(f"Median logP mismatch count: {median_mismatch_count}")


def clean_value(x):
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def unique_join(series):
    vals = []
    for x in series:
        sx = clean_value(x)
        if sx and sx not in vals:
            vals.append(sx)
    return "; ".join(vals)


def make_source_reference(group):
    refs = []
    for _, row in group.iterrows():
        authors = clean_value(row.get("Authors", ""))
        year = clean_value(row.get("Year", ""))
        journal = clean_value(row.get("Journal", ""))
        doi = clean_value(row.get("Data_Source_DOI", ""))

        parts = []
        if authors:
            parts.append(authors)
        if year:
            parts.append(year)
        if journal:
            parts.append(journal)
        if doi:
            parts.append("DOI: " + doi)

        ref = ", ".join(parts)
        if ref and ref not in refs:
            refs.append(ref)

    return " | ".join(refs)


rows = []

for cid in list(s1["Compound_ID"]):
    g = raw[raw["Compound_ID"] == cid].copy()
    s1row = s1[s1["Compound_ID"] == cid].iloc[0]

    accepted_values = []
    for _, r in g.iterrows():
        rid = clean_value(r.get("Replicate_ID", ""))
        val = clean_value(r.get("Experimental_logP", ""))
        if rid:
            accepted_values.append(f"{rid}: {val}")
        else:
            accepted_values.append(val)

    if len(g) > 1:
        note = (
            "Multiple accepted source/replicate records were consolidated by "
            "median Experimental_logP; median value matches Dataset_S1 logP_exp."
        )
    else:
        note = (
            "Single accepted experimental record; value matches Dataset_S1 logP_exp."
        )

    raw_notes = unique_join(g["Notes"]) if "Notes" in g.columns else ""
    if raw_notes:
        note = note + " Raw notes: " + raw_notes

    rows.append({
        "Compound_ID": cid,
        "SMILES": s1row["SMILES"],
        "logP_exp": s1row["logP_exp"],
        "n_replicates": int(s1row["n_replicates"]),
        "Source_record_count": int(len(g)),
        "Accepted_experimental_logP_values": "; ".join(accepted_values),
        "Source_reference": make_source_reference(g),
        "Measurement_method": unique_join(g["Measurement_Method"]),
        "Experimental_conditions": unique_join(g["Experimental_Conditions"]),
        "Data_Tier": s1row["Data_Tier"],
        "DOI": unique_join(g["Data_Source_DOI"]),
        "Journal": unique_join(g["Journal"]),
        "Year": unique_join(g["Year"]),
        "Authors": unique_join(g["Authors"]),
        "Curation_notes": note,
    })

s2 = pd.DataFrame(rows)
s2.to_csv(out_s2, index=False, encoding="utf-8-sig")

s3 = pd.DataFrame([{
    "Audit_item": "compound_level_exclusion_check",
    "Raw_file": "data/raw/Dataset_S0_raw_literature_collection.xlsx",
    "Final_dataset": "data/processed/Dataset_S1_benchmark_dataset.csv",
    "Raw_source_level_rows": int(len(raw)),
    "Raw_unique_compounds": int(raw_unique),
    "Final_compound_level_rows": int(len(s1)),
    "Final_unique_compounds": int(s1_unique),
    "Raw_IDs_not_in_final": "none",
    "Final_IDs_not_in_raw": "none",
    "Median_logP_mismatch_count": int(median_mismatch_count),
    "Curation_conclusion": (
        "No compound-level records were excluded between the frozen raw archive "
        "and final benchmark dataset. The 100 raw source/replicate-level records "
        "were consolidated into 95 compound-level benchmark records by median "
        "Experimental_logP where replicate records were available."
    ),
    "Recommended_SI_interpretation": (
        "This file is a curation/exclusion audit rather than a list of excluded compounds."
    ),
}])

s3.to_csv(out_s3, index=False, encoding="utf-8-sig")

print("WROTE:", out_s2)
print("S2 shape:", s2.shape)
print("WROTE:", out_s3)
print("S3 shape:", s3.shape)
print("Validation passed: Compound_ID sets match and median logP values match exactly.")
print("S2 Data_Tier counts:")
print(s2["Data_Tier"].value_counts().to_string())