# -*- coding: utf-8 -*-
"""
Generate publication-clean Figures 6–8 for the coumarin-logp manuscript.

Project structure:
coumarin-logp/
│
├── data/
│   └── processed/
│       └── benchmark_dataset.csv
│
├── scripts/
│   └── Figure_6_8_clean.py
│
└── figures/
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 0. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

dataset_path = PROJECT_DIR / "data" / "processed" / "benchmark_dataset.csv"
outdir = PROJECT_DIR / "figures"
outdir.mkdir(parents=True, exist_ok=True)

if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset bulunamadı: {dataset_path}")

print(f"Reading dataset from: {dataset_path}")
print(f"Saving figures to:   {outdir}")


# ============================================================
# 1. Global plot settings
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)
    ax.grid(False)


# ============================================================
# 2. Load dataset and prepare values
# ============================================================

df = pd.read_csv(dataset_path)

required_columns = [
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos_IT",
    "delta_Consensus",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Dataset içinde eksik kolon(lar) var: {missing}")

fragment_cols = ["XLOGP3", "WLOGP", "MLOGP", "Silicos_IT"]

df["Spread4"] = df[fragment_cols].max(axis=1) - df[fragment_cols].min(axis=1)
df["abs_consensus_error"] = df["delta_Consensus"].abs()
df["severe_error"] = np.where(
    df["abs_consensus_error"] >= 2.0,
    "|ΔConsensus| ≥ 2.0",
    "|ΔConsensus| < 2.0",
)

pearson_r, pearson_p = stats.pearsonr(df["Spread4"], df["abs_consensus_error"])
spearman_r, spearman_p = stats.spearmanr(df["Spread4"], df["abs_consensus_error"])

spread_low = df.loc[df["severe_error"] == "|ΔConsensus| < 2.0", "Spread4"]
spread_high = df.loc[df["severe_error"] == "|ΔConsensus| ≥ 2.0", "Spread4"]

u_res = stats.mannwhitneyu(spread_high, spread_low, alternative="two-sided")


# ============================================================
# 3. Figure 6 — two-panel clean layout
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

# ---- Panel A
ax = axes[0]

low_group = df[df["severe_error"] == "|ΔConsensus| < 2.0"]
high_group = df[df["severe_error"] == "|ΔConsensus| ≥ 2.0"]

ax.scatter(
    low_group["Spread4"],
    low_group["abs_consensus_error"],
    s=55,
    alpha=0.8,
    label="|ΔConsensus| < 2.0",
)

ax.scatter(
    high_group["Spread4"],
    high_group["abs_consensus_error"],
    s=75,
    alpha=0.9,
    label="|ΔConsensus| ≥ 2.0",
)

# regression line
x = df["Spread4"].to_numpy()
y = df["abs_consensus_error"].to_numpy()
slope, intercept = np.polyfit(x, y, 1)
xline = np.linspace(x.min(), x.max(), 300)
ax.plot(xline, slope * xline + intercept, linewidth=2.0)

ax.set_title("A. Predictor disagreement versus absolute consensus error", pad=10)
ax.set_xlabel("Spread4")
ax.set_ylabel("|ΔConsensus| (log units)")

ax.text(
    0.03,
    0.97,
    f"Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}\np ≈ 10⁻¹²",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
)

ax.legend(frameon=False, loc="lower right")
style_axes(ax)

# ---- Panel B
ax = axes[1]

box = ax.boxplot(
    [spread_low, spread_high],
    widths=0.28,
    patch_artist=True,
    labels=["|ΔConsensus| < 2.0", "|ΔConsensus| ≥ 2.0"],
    medianprops=dict(linewidth=1.8, color="black"),
    boxprops=dict(linewidth=1.4),
    whiskerprops=dict(linewidth=1.3),
    capprops=dict(linewidth=1.3),
)

# light fills
fills = ["#d9ecff", "#ffe2cc"]
for patch, c in zip(box["boxes"], fills):
    patch.set_facecolor(c)
    patch.set_edgecolor("black")

rng = np.random.default_rng(42)

x1 = 1 + rng.normal(0, 0.03, size=len(spread_low))
x2 = 2 + rng.normal(0, 0.03, size=len(spread_high))

ax.scatter(x1, spread_low, s=36, alpha=0.65)
ax.scatter(x2, spread_high, s=46, alpha=0.8)

ax.set_title("B. Predictor disagreement by severe-error class", pad=10)
ax.set_ylabel("Spread4")
ax.text(
    0.03,
    0.97,
    "Mann–Whitney p = 4.82 × 10⁻⁴",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
)

style_axes(ax)

fig6_png = outdir / "Figure_6_clean.png"
fig6_svg = outdir / "Figure_6_clean.svg"
fig.savefig(fig6_png, dpi=600, bbox_inches="tight")
fig.savefig(fig6_svg, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 4. Figure 7 — clean DFT contrast plot
# ============================================================

dft = pd.DataFrame({
    "Compound": ["CMR_055", "CMR_043", "CMR_079", "CMR_058"],
    "Gap": [4.6161, 4.1328, 1.1528, 0.3872],
    "Dipole": [4.8037, 5.8138, 8.1159, 19.4203],
    "Role": [
        "N-free reference",
        "Accurate N = 2 control",
        "Multi-N low-error reference",
        "D–π–A failure case",
    ]
})

fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

sizes = [110, 110, 110, 130]
colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

for i, row in dft.iterrows():
    ax.scatter(row["Gap"], row["Dipole"], s=sizes[i], color=colors[i], edgecolor="black", linewidth=0.5, zorder=3)

# carefully positioned labels
offsets = {
    "CMR_055": (-58, 10),
    "CMR_043": (10, 8),
    "CMR_079": (8, 10),
    "CMR_058": (8, 8),
}

for _, row in dft.iterrows():
    dx, dy = offsets[row["Compound"]]
    ax.annotate(
        row["Compound"],
        (row["Gap"], row["Dipole"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=10,
    )

ax.set_title("Electronic contrast across the mechanistic DFT panel", pad=10)
ax.set_xlabel("HOMO–LUMO gap (eV)")
ax.set_ylabel("Dipole moment (D)")

# expand limits to avoid crowding
ax.set_xlim(0.0, 5.0)
ax.set_ylim(4.0, 20.5)

style_axes(ax)

fig7_png = outdir / "Figure_7_clean.png"
fig7_svg = outdir / "Figure_7_clean.svg"
fig.savefig(fig7_png, dpi=600, bbox_inches="tight")
fig.savefig(fig7_svg, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 5. Figure 8 — cleaner failure-mode taxonomy
# ============================================================

fig, ax = plt.subplots(figsize=(8.4, 5.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(
    0.5, 0.96,
    "Failure-mode taxonomy for SwissADME-associated logP prediction in coumarin derivatives",
    ha="center", va="top",
    fontsize=14, fontweight="bold"
)

# box content
boxes = {
    "FM0": (0.24, 0.78, "FM0\nLow-risk / serviceable\nFragment locality retained\nUse cautiously if predictors agree"),
    "FM1": (0.74, 0.78, "FM1\nPolar overestimation\nN = 1–3, logP < 1.5\nPrioritise experimental logP"),
    "FM2": (0.24, 0.50, "FM2\nConjugated-N misassignment\nN = 1–3, logP > 1.5\nTreat predictions as provisional"),
    "FM3": (0.74, 0.50, "FM3\nHigh-N cancellation\nN ≥ 4, mixed N environments\nInspect structure individually"),
    "FM4": (0.49, 0.20, "FM4\nConjugation overflow\nN = 0, logP > 3.0\nWatch for underestimation"),
}

box_w = 0.26
box_h = 0.12

for key, (x, y, text) in boxes.items():
    ax.text(
        x, y, text,
        ha="center", va="center", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="black", lw=1.4)
    )

arrowprops = dict(arrowstyle="->", lw=1.4, color="black")

# arrows
ax.annotate("", xy=(0.61, 0.78), xytext=(0.37, 0.78), arrowprops=arrowprops)
ax.annotate("", xy=(0.24, 0.60), xytext=(0.24, 0.71), arrowprops=arrowprops)
ax.annotate("", xy=(0.74, 0.60), xytext=(0.74, 0.71), arrowprops=arrowprops)
ax.annotate("", xy=(0.44, 0.27), xytext=(0.30, 0.41), arrowprops=arrowprops)
ax.annotate("", xy=(0.54, 0.27), xytext=(0.68, 0.41), arrowprops=arrowprops)

ax.text(
    0.5, 0.055,
    "Risk increases when fragment locality is lost through conjugation, nitrogen embedding,\n"
    "donor–acceptor character, and/or high predictor disagreement.",
    ha="center", va="center", fontsize=11
)

fig.tight_layout()
fig8_png = outdir / "Figure_8_clean.png"
fig8_svg = outdir / "Figure_8_clean.svg"
fig.savefig(fig8_png, dpi=600, bbox_inches="tight")
fig.savefig(fig8_svg, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 6. Console summary
# ============================================================

print("\nCreated clean figure files:")
print(f"- {fig6_png}")
print(f"- {fig6_svg}")
print(f"- {fig7_png}")
print(f"- {fig7_svg}")
print(f"- {fig8_png}")
print(f"- {fig8_svg}")

print("\nCheck values:")
print(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.2e}")
print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.2e}")
print(f"Mann–Whitney p = {u_res.pvalue:.2e}")