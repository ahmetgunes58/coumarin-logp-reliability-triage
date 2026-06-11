# -*- coding: utf-8 -*-
"""
Figure 4
Structural-class dependence of consensus logP prediction error
Final main-text version

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python scripts\26_make_figure_4_structural_class_error.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

OUT_FIG_DIR = PROJECT_DIR / "figures" / "main"
OUT_SRC_DIR = PROJECT_DIR / "figures" / "source_data"
OUT_DATA_DIR = PROJECT_DIR / "data" / "processed"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SRC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_FIG_DIR / "Figure_4_structural_class_consensus_error.png"
OUT_PDF = OUT_FIG_DIR / "Figure_4_structural_class_consensus_error.pdf"
OUT_SVG = OUT_FIG_DIR / "Figure_4_structural_class_consensus_error.svg"
OUT_TIF = OUT_FIG_DIR / "Figure_4_structural_class_consensus_error.tif"

SOURCE_CSV = OUT_SRC_DIR / "Figure_4_structural_class_consensus_error_source_data.csv"
PROCESSED_CSV = OUT_DATA_DIR / "Dataset_Figure_4_structural_class_consensus_error.csv"


# ============================================================
# 2. Structural-class summary data
# ============================================================
# Values aligned with manuscript Table 3.

rows = [
    {
        "Structural_class": "Oxadiazole",
        "Short_label": "Oxadiazole",
        "n": 5,
        "MAE": 0.13,
        "RMSE": 0.16,
        "Bias": -0.06,
        "Interpretation": "Compact / electronically contained N",
    },
    {
        "Structural_class": "Furocoumarin",
        "Short_label": "Furocoumarin",
        "n": 4,
        "MAE": 0.22,
        "RMSE": 0.27,
        "Bias": -0.22,
        "Interpretation": "Fused O-ring, N-free",
    },
    {
        "Structural_class": "Oxadiazoline",
        "Short_label": "Oxadiazoline",
        "n": 3,
        "MAE": 0.35,
        "RMSE": 0.37,
        "Bias": -0.23,
        "Interpretation": "Compact reduced N-heterocycle",
    },
    {
        "Structural_class": "Amidoxime",
        "Short_label": "Amidoxime",
        "n": 3,
        "MAE": 0.64,
        "RMSE": 0.64,
        "Bias": -0.64,
        "Interpretation": "Moderate N-heteroatom polarity",
    },
    {
        "Structural_class": "Simple coumarin",
        "Short_label": "Simple coumarin",
        "n": 3,
        "MAE": 0.69,
        "RMSE": 0.79,
        "Bias": +0.69,
        "Interpretation": "N-free, short π-system",
    },
    {
        "Structural_class": "Low-frequency / singleton classes",
        "Short_label": "Low-frequency /\nsingleton classes",
        "n": 10,
        "MAE": 0.66,
        "RMSE": 0.81,
        "Bias": -0.42,
        "Interpretation": "Mixed low-frequency structural types",
    },
    {
        "Structural_class": "Triazolo-thiadiazinyl",
        "Short_label": "Triazolo-thiadiazinyl",
        "n": 10,
        "MAE": 1.27,
        "RMSE": 1.33,
        "Bias": -1.27,
        "Interpretation": "Multi-N compact fused heteroaromatic system",
    },
    {
        "Structural_class": "Conjugated N-bearing coumarin",
        "Short_label": "Conjugated\nN-bearing coumarin",
        "n": 47,
        "MAE": 1.33,
        "RMSE": 1.63,
        "Bias": -1.08,
        "Interpretation": "Extended π-system with nitrogen-containing substituents",
    },
    {
        "Structural_class": "Phosphonate coumarin",
        "Short_label": "Phosphonate coumarin",
        "n": 5,
        "MAE": 1.65,
        "RMSE": 1.72,
        "Bias": -1.65,
        "Interpretation": "N / phosphonate-containing polar conjugated system",
    },
    {
        "Structural_class": "Dimeric coumarin",
        "Short_label": "Dimeric coumarin",
        "n": 3,
        "MAE": 1.82,
        "RMSE": 1.87,
        "Bias": +0.99,
        "Interpretation": "N-free, extended π-conjugated dimer",
    },
    {
        "Structural_class": "7-amino donor–acceptor-conjugated",
        "Short_label": "7-amino donor–acceptor-\nconjugated",
        "n": 2,
        "MAE": 4.35,
        "RMSE": 4.43,
        "Bias": -4.35,
        "Interpretation": "Amino-donor/coumarin-lactone acceptor conjugation; extreme-case class",
    },
]

df = pd.DataFrame(rows)

# Sort by MAE ascending for visual gradient
df = df.sort_values("MAE", ascending=True).reset_index(drop=True)

df.to_csv(SOURCE_CSV, index=False, encoding="utf-8-sig")
df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 3. Figure style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 4. Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8.8, 5.8))

y = np.arange(len(df))

# MAE-based muted gradient; bar length already encodes MAE, so colors are subtle.
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(df)))

bars = ax.barh(
    y,
    df["MAE"],
    color=colors,
    edgecolor="black",
    linewidth=0.45,
    height=0.62,
)

# Emphasise extreme-case class with a slightly darker edge
for idx, row in df.iterrows():
    if row["Structural_class"] == "7-amino donor–acceptor-conjugated":
        bars[idx].set_edgecolor("black")
        bars[idx].set_linewidth(1.1)

# Vertical zero line
ax.axvline(0, color="black", linewidth=0.9)

# Optional light reference at MAE = 1.0 log unit
ax.axvline(
    1.0,
    color="0.55",
    linestyle="--",
    linewidth=0.8,
    alpha=0.8,
)
ax.text(
    1.02,
    -0.72,
    "MAE = 1.0",
    ha="left",
    va="center",
    fontsize=8.5,
    color="0.35",
)

# Bar-end annotations: n and mean bias
for i, row in df.iterrows():
    x = row["MAE"]
    annotation = f"n={int(row['n'])}; Δ={row['Bias']:+.2f}"
    ax.text(
        x + 0.06,
        i,
        annotation,
        va="center",
        ha="left",
        fontsize=8.6,
        color="0.15",
    )

# Axis labels
ax.set_yticks(y)
ax.set_yticklabels(df["Short_label"])

ax.set_xlabel("Consensus-model MAE (log units)")
ax.set_ylabel("Structural class")

# Limits
ax.set_xlim(0, max(df["MAE"]) + 0.75)
ax.set_ylim(-0.8, len(df) - 0.2)

# Grid
ax.xaxis.grid(True, linestyle="-", linewidth=0.45, alpha=0.35)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

# Remove unnecessary spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Invert y-axis so low-error classes appear at top
ax.invert_yaxis()

# Compact note inside the figure area
note = (
    # "All structural classes covering the 95-compound dataset are shown. "
    # "Bias = experimental logP − predicted consensus logP."
)



plt.tight_layout()


# ============================================================
# 5. Save
# ============================================================

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")
fig.savefig(OUT_TIF, dpi=600, bbox_inches="tight")

plt.close(fig)

print("Figure 4 generated successfully.")
print(f"PNG:        {OUT_PNG}")
print(f"PDF:        {OUT_PDF}")
print(f"SVG:        {OUT_SVG}")
print(f"TIF:        {OUT_TIF}")
print(f"Source CSV: {SOURCE_CSV}")
print(f"Data CSV:   {PROCESSED_CSV}")