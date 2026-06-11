# -*- coding: utf-8 -*-
"""
Figure S2. Signed prediction-error distributions for all six SwissADME-associated predictors.
Figure S3. Absolute prediction-error distributions for all six SwissADME-associated predictors.

Input:
    coumarin-logp/data/processed/Dataset_S1_benchmark_dataset.csv

Outputs:
    coumarin-logp/data/processed/Dataset_S18_six_predictor_signed_error_source_data.csv
    coumarin-logp/data/processed/Dataset_S19_six_predictor_absolute_error_source_data.csv

    coumarin-logp/figures/supporting/Figure_S2_six_predictor_signed_error_distributions.png
    coumarin-logp/figures/supporting/Figure_S2_six_predictor_signed_error_distributions.tiff
    coumarin-logp/figures/supporting/Figure_S2_six_predictor_signed_error_distributions.pdf

    coumarin-logp/figures/supporting/Figure_S3_six_predictor_absolute_error_distributions.png
    coumarin-logp/figures/supporting/Figure_S3_six_predictor_absolute_error_distributions.tiff
    coumarin-logp/figures/supporting/Figure_S3_six_predictor_absolute_error_distributions.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")

INPUT_CSV = ROOT / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "supporting"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIGNED_SOURCE = DATA_DIR / "Dataset_S18_six_predictor_signed_error_source_data.csv"
ABS_SOURCE = DATA_DIR / "Dataset_S19_six_predictor_absolute_error_source_data.csv"

S2_PNG = FIG_DIR / "Figure_S2_six_predictor_signed_error_distributions.png"
S2_TIFF = FIG_DIR / "Figure_S2_six_predictor_signed_error_distributions.tiff"
S2_PDF = FIG_DIR / "Figure_S2_six_predictor_signed_error_distributions.pdf"

S3_PNG = FIG_DIR / "Figure_S3_six_predictor_absolute_error_distributions.png"
S3_TIFF = FIG_DIR / "Figure_S3_six_predictor_absolute_error_distributions.tiff"
S3_PDF = FIG_DIR / "Figure_S3_six_predictor_absolute_error_distributions.pdf"


# ============================================================
# Configuration
# ============================================================

predictor_order = [
    ("iLOGP", "iLOGP"),
    ("XLOGP3", "XLOGP3"),
    ("WLOGP", "WLOGP"),
    ("MLOGP", "MLOGP"),
    ("Silicos_IT", "Silicos-IT"),
    ("Consensus", "Consensus"),
]

required_cols = ["Compound_ID", "logP_exp"] + [p[0] for p in predictor_order]


# ============================================================
# Helper
# ============================================================

def make_long_error_table(df, mode="signed"):
    rows = []
    for pred_col, pred_label in predictor_order:
        for _, row in df.iterrows():
            logp_exp = row["logP_exp"]
            logp_pred = row[pred_col]
            delta = logp_exp - logp_pred
            abs_delta = abs(delta)

            rows.append({
                "Compound_ID": row["Compound_ID"],
                "Predictor": pred_label,
                "logP_exp": logp_exp,
                "logP_pred": logp_pred,
                "Delta_logP": delta,
                "Abs_Delta_logP": abs_delta,
                "Plot_value": delta if mode == "signed" else abs_delta
            })
    return pd.DataFrame(rows)


def add_jittered_points(ax, grouped_data, positions, seed=42):
    rng = np.random.default_rng(seed)
    for vals, pos in zip(grouped_data, positions):
        x = rng.normal(loc=pos, scale=0.06, size=len(vals))
        ax.scatter(
            x,
            vals,
            s=12,
            alpha=0.55,
            linewidths=0.3,
            edgecolors="black"
        )


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Eksik sütun(lar): {missing_cols}")

for col in required_cols[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_cols).copy()

n_compounds = len(df)
if n_compounds != 95:
    print(f"Uyarı: Beklenen n = 95, bulunan n = {n_compounds}")


# ============================================================
# Build source data
# ============================================================

signed_df = make_long_error_table(df, mode="signed")
abs_df = make_long_error_table(df, mode="absolute")

signed_df.to_csv(SIGNED_SOURCE, index=False, encoding="utf-8-sig")
abs_df.to_csv(ABS_SOURCE, index=False, encoding="utf-8-sig")


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ============================================================
# Figure S2: signed error distributions
# ============================================================

fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=600)

signed_groups = [
    signed_df.loc[signed_df["Predictor"] == label, "Delta_logP"].values
    for _, label in predictor_order
]
signed_labels = [label for _, label in predictor_order]
positions = np.arange(1, len(signed_labels) + 1)

ax.boxplot(
    signed_groups,
    positions=positions,
    widths=0.55,
    patch_artist=False,
    showfliers=False,
    medianprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=0.9),
    capprops=dict(linewidth=0.9),
    boxprops=dict(linewidth=0.9),
)

add_jittered_points(ax, signed_groups, positions, seed=42)

ax.axhline(0, linestyle="--", linewidth=1.0, color="black")
ax.set_xticks(positions)
ax.set_xticklabels(signed_labels, rotation=20, ha="right")
ax.set_ylabel("Signed error, ΔlogP")
ax.set_xlabel("")
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(S2_PNG, dpi=600, bbox_inches="tight")
fig.savefig(S2_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(S2_PDF, bbox_inches="tight")

plt.close(fig)


# ============================================================
# Figure S3: absolute error distributions
# ============================================================

fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=600)

abs_groups = [
    abs_df.loc[abs_df["Predictor"] == label, "Abs_Delta_logP"].values
    for _, label in predictor_order
]

ax.boxplot(
    abs_groups,
    positions=positions,
    widths=0.55,
    patch_artist=False,
    showfliers=False,
    medianprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=0.9),
    capprops=dict(linewidth=0.9),
    boxprops=dict(linewidth=0.9),
)

add_jittered_points(ax, abs_groups, positions, seed=84)

ax.set_xticks(positions)
ax.set_xticklabels(signed_labels, rotation=20, ha="right")
ax.set_ylabel("Absolute error, |ΔlogP|")
ax.set_xlabel("")
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(S3_PNG, dpi=600, bbox_inches="tight")
fig.savefig(S3_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(S3_PDF, bbox_inches="tight")

plt.close(fig)


# ============================================================
# Final print
# ============================================================

print("Figure S2 and Figure S3 başarıyla oluşturuldu.")
print(f"Girdi dosyası        : {INPUT_CSV}")
print(f"Kullanılan bileşik n : {n_compounds}")
print("")
print("Kaynak veri dosyaları:")
print(f"  {SIGNED_SOURCE}")
print(f"  {ABS_SOURCE}")
print("")
print("Figure S2 çıktıları:")
print(f"  {S2_PNG}")
print(f"  {S2_TIFF}")
print(f"  {S2_PDF}")
print("")
print("Figure S3 çıktıları:")
print(f"  {S3_PNG}")
print(f"  {S3_TIFF}")
print(f"  {S3_PDF}")