# -*- coding: utf-8 -*-
"""
Figure 6. Predictor disagreement identifies compounds in a non-additive prediction regime.

Updated version:
- white background
- better p-value placement in panel B
- scientific p-value formatting (e.g., 4.82 × 10^-4)
- improved annotation placement for CMR_GOLD_058
- severe-error points highlighted separately in panel A
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu


# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")

INPUT_CSV = ROOT / "data" / "processed" / "benchmark_dataset.csv"

OUTPUT_DIR = ROOT / "figures" / "Figure6_predictor_disagreement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUTPUT_DIR / "Figure6_predictor_disagreement_v2.png"
FIG_TIFF = OUTPUT_DIR / "Figure6_predictor_disagreement_v2.tiff"
FIG_PDF = OUTPUT_DIR / "Figure6_predictor_disagreement_v2.pdf"
SOURCE_CSV = OUTPUT_DIR / "Figure6_source_data_v2.csv"
STATS_TXT = OUTPUT_DIR / "Figure6_statistics_v2.txt"


# ============================================================
# Helpers
# ============================================================

def format_p_mathtext(p):
    """
    Format p-value as mathtext:
    4.82 × 10^-4  -> r"$4.82 \times 10^{-4}$"
    """
    if p == 0:
        return r"$< 10^{-300}$"

    exponent = int(np.floor(np.log10(abs(p))))
    mantissa = p / (10 ** exponent)

    if exponent in [0, 1, 2]:
        return f"{p:.3f}"

    return rf"${mantissa:.2f} \times 10^{{{exponent}}}$"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_cols = [
    "Compound_ID",
    "logP_exp",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos_IT",
    "Consensus",
    "delta_Consensus",
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

numeric_cols = [
    "logP_exp",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos_IT",
    "Consensus",
    "delta_Consensus",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols).copy()


# ============================================================
# Calculate Figure 6 variables
# ============================================================

fragment_predictors = ["XLOGP3", "WLOGP", "MLOGP", "Silicos_IT"]

df["Spread4"] = (
    df[fragment_predictors].max(axis=1)
    - df[fragment_predictors].min(axis=1)
)

df["Consensus_Error_Check"] = df["logP_exp"] - df["Consensus"]
df["Delta_Difference"] = df["delta_Consensus"] - df["Consensus_Error_Check"]
max_delta_diff = df["Delta_Difference"].abs().max()

df["Consensus_Error"] = df["delta_Consensus"]
df["Abs_Consensus_Error"] = df["Consensus_Error"].abs()

df["Severe_Error"] = df["Abs_Consensus_Error"] >= 2.0

df["Severe_Error_Group"] = np.where(
    df["Severe_Error"],
    "|consensus error| ≥ 2.0",
    "|consensus error| < 2.0",
)


# ============================================================
# Statistics
# ============================================================

pearson_r, pearson_p = pearsonr(df["Spread4"], df["Abs_Consensus_Error"])
spearman_rho, spearman_p = spearmanr(df["Spread4"], df["Abs_Consensus_Error"])

severe = df.loc[df["Severe_Error"], "Spread4"]
non_severe = df.loc[~df["Severe_Error"], "Spread4"]

mw_u, mw_p = mannwhitneyu(
    severe,
    non_severe,
    alternative="two-sided"
)

summary = {
    "n_total": len(df),
    "n_severe_abs_error_ge_2": len(severe),
    "n_non_severe_abs_error_lt_2": len(non_severe),
    "pearson_r": pearson_r,
    "pearson_p": pearson_p,
    "spearman_rho": spearman_rho,
    "spearman_p": spearman_p,
    "mean_spread4_severe": severe.mean(),
    "median_spread4_severe": severe.median(),
    "mean_spread4_non_severe": non_severe.mean(),
    "median_spread4_non_severe": non_severe.median(),
    "mannwhitney_u": mw_u,
    "mannwhitney_p": mw_p,
    "max_delta_consensus_consistency_difference": max_delta_diff,
}


# ============================================================
# Save source data and statistics
# ============================================================

source_cols = [
    "Compound_ID",
    "logP_exp",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos_IT",
    "Consensus",
    "Consensus_Error",
    "Abs_Consensus_Error",
    "Spread4",
    "Severe_Error_Group",
]

df[source_cols].to_csv(SOURCE_CSV, index=False, encoding="utf-8-sig")

with open(STATS_TXT, "w", encoding="utf-8") as f:
    f.write("Figure 6 statistics\n")
    f.write("===================\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
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


fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.2),
    dpi=600,
    constrained_layout=True,
)

fig.patch.set_facecolor("white")

ax1, ax2 = axes


# ============================================================
# Panel A
# ============================================================

df_non = df[~df["Severe_Error"]].copy()
df_sev = df[df["Severe_Error"]].copy()

ax1.scatter(
    df_non["Spread4"],
    df_non["Abs_Consensus_Error"],
    s=28,
    alpha=0.75,
    linewidths=0,
    color="#4C9BD4",
    label="|error| < 2.0",
)

ax1.scatter(
    df_sev["Spread4"],
    df_sev["Abs_Consensus_Error"],
    s=32,
    alpha=0.85,
    linewidths=0,
    color="#F28E2B",
    label="|error| ≥ 2.0",
)

# Linear guide line
x = df["Spread4"].to_numpy()
y = df["Abs_Consensus_Error"].to_numpy()

coef = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = coef[0] * x_line + coef[1]

ax1.plot(
    x_line,
    y_line,
    linewidth=1.3,
    color="#2C7FB8",
)

# severe-error threshold
ax1.axhline(
    2.0,
    linestyle="--",
    linewidth=0.9,
    alpha=0.9,
    color="#4C9BD4",
)

ax1.set_xlabel("Predictor disagreement, Spread4")
ax1.set_ylabel("|Consensus error| (log units)")
ax1.set_title("A", loc="left", fontweight="bold", fontsize=11)

ax1.text(
    0.04,
    0.96,
    f"Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_rho:.3f}",
    transform=ax1.transAxes,
    va="top",
    ha="left",
)

# Better placement for CMR_GOLD_058
target_id = "CMR_GOLD_058"
target_rows = df[df["Compound_ID"].astype(str) == target_id]

if not target_rows.empty:
    row = target_rows.iloc[0]
    ax1.annotate(
        target_id,
        xy=(row["Spread4"], row["Abs_Consensus_Error"]),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=7,
        ha="left",
        va="bottom",
    )

ax1.legend(
    frameon=False,
    fontsize=7,
    loc="lower right",
)

# nicer y-limits
ax1.set_ylim(bottom=-0.1, top=max(df["Abs_Consensus_Error"].max() + 0.35, 5.4))


# ============================================================
# Panel B
# ============================================================

data_groups = [
    non_severe.to_numpy(),
    severe.to_numpy(),
]

positions = [1, 2]

box = ax2.boxplot(
    data_groups,
    positions=positions,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(linewidth=1.2, color="black"),
    boxprops=dict(linewidth=1.0, edgecolor="black"),
    whiskerprops=dict(linewidth=1.0, color="black"),
    capprops=dict(linewidth=1.0, color="black"),
)

# Fill box colors
box["boxes"][0].set_facecolor("#DCEAF7")
box["boxes"][1].set_facecolor("#FBE5CF")

# Individual points
rng = np.random.default_rng(42)

group_colors = ["#4C9BD4", "#F28E2B"]

for pos, values, color in zip(positions, data_groups, group_colors):
    jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
    ax2.scatter(
        np.full(len(values), pos) + jitter,
        values,
        s=22,
        alpha=0.82,
        linewidths=0,
        color=color,
        zorder=3,
    )

ax2.set_xticks(positions)
ax2.set_xticklabels(
    [
        f"|error| < 2.0\n(n = {len(non_severe)})",
        f"|error| ≥ 2.0\n(n = {len(severe)})",
    ]
)

ax2.set_ylabel("Predictor disagreement, Spread4")
ax2.set_title("B", loc="left", fontweight="bold", fontsize=11)

# Dynamic y-range and p-value placement
data_max = max(non_severe.max(), severe.max())
data_min = min(non_severe.min(), severe.min())

y_bracket = data_max + 0.12
y_text = data_max + 0.28
y_top = data_max + 0.45

# Significance bracket
ax2.plot(
    [1, 1, 2, 2],
    [y_bracket - 0.03, y_bracket, y_bracket, y_bracket - 0.03],
    color="black",
    linewidth=1.0,
)

ax2.text(
    1.5,
    y_text,
    rf"Mann–Whitney p = {format_p_mathtext(mw_p)}",
    ha="center",
    va="bottom",
)

ax2.set_ylim(bottom=max(0, data_min - 0.2), top=y_top)


# ============================================================
# Clean axes
# ============================================================

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


# ============================================================
# Export
# ============================================================

fig.savefig(FIG_PNG, dpi=600, bbox_inches="tight")
fig.savefig(FIG_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Console output
# ============================================================

print("\nFigure 6 generated successfully.")
print(f"Input:       {INPUT_CSV}")
print(f"PNG:         {FIG_PNG}")
print(f"TIFF:        {FIG_TIFF}")
print(f"PDF:         {FIG_PDF}")
print(f"Source data: {SOURCE_CSV}")
print(f"Stats:       {STATS_TXT}")

print("\nKey statistics:")
for key, value in summary.items():
    print(f"{key}: {value}")