# -*- coding: utf-8 -*-
"""
Figure 3
Consensus logP error distribution and nitrogen-count-group displacement
Final two-panel main-text version

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python scripts\27_make_figure_3_consensus_error_distribution.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

OUT_FIG_DIR = PROJECT_DIR / "figures" / "main"
OUT_SRC_DIR = PROJECT_DIR / "figures" / "source_data"
OUT_DATA_DIR = PROJECT_DIR / "data" / "processed"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SRC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_FIG_DIR / "Figure_3_consensus_error_distribution.png"
OUT_PDF = OUT_FIG_DIR / "Figure_3_consensus_error_distribution.pdf"
OUT_SVG = OUT_FIG_DIR / "Figure_3_consensus_error_distribution.svg"
OUT_TIF = OUT_FIG_DIR / "Figure_3_consensus_error_distribution.tif"

SOURCE_CSV = OUT_SRC_DIR / "Figure_3_consensus_error_distribution_source_data.csv"
STATS_TXT = OUT_DATA_DIR / "Dataset_Figure_3_consensus_error_distribution_statistics.txt"


# ============================================================
# 2. Helpers
# ============================================================

def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {
        c.lower().replace(" ", "").replace("-", "_"): c
        for c in df.columns
    }
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("-", "_")
        if key in normalized:
            return normalized[key]
    raise KeyError(
        f"None of the candidate columns were found: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def n_group_from_count(n: int) -> str:
    if n == 0:
        return "N = 0"
    if n == 1:
        return "N = 1"
    if n in (2, 3):
        return "N = 2–3"
    return "N ≥ 4"


# ============================================================
# 3. Load data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

compound_col = find_column(df, ["Compound_ID", "compound_id", "ID"])
n_count_col = find_column(df, ["N_count", "n_count", "Nitrogen_count"])

try:
    delta_col = find_column(
        df,
        ["delta_Consensus", "Delta_Consensus", "delta_consensus", "Consensus_delta"]
    )
    work = df[[compound_col, n_count_col, delta_col]].copy()
    work.columns = ["Compound_ID", "N_count", "delta_Consensus"]
except KeyError:
    logp_col = find_column(df, ["logP_exp", "LogP_exp", "experimental_logP", "Exp_logP"])
    consensus_col = find_column(df, ["Consensus", "consensus", "Consensus_logP", "consensus_logP"])
    work = df[[compound_col, n_count_col, logp_col, consensus_col]].copy()
    work.columns = ["Compound_ID", "N_count", "logP_exp", "Consensus"]
    work["delta_Consensus"] = work["logP_exp"] - work["Consensus"]

work["N_count"] = work["N_count"].astype(int)
work["N_group"] = work["N_count"].apply(n_group_from_count)
work["abs_delta_Consensus"] = work["delta_Consensus"].abs()
work["Overestimated"] = work["delta_Consensus"] < 0
work["Severe_error"] = work["abs_delta_Consensus"] >= 2.0

group_order = ["N = 0", "N = 1", "N = 2–3", "N ≥ 4"]

work.to_csv(SOURCE_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 4. Statistics
# ============================================================

errors = work["delta_Consensus"].dropna().astype(float)

mean_error = float(errors.mean())
median_error = float(errors.median())
std_error = float(errors.std(ddof=1))
overestimated_percent = float((errors < 0).mean() * 100)
abs_gt_1_percent = float((errors.abs() > 1.0).mean() * 100)
abs_ge_2_percent = float((errors.abs() >= 2.0).mean() * 100)

shapiro_w, shapiro_p = stats.shapiro(errors)

group_summary = (
    work.groupby("N_group")
    .agg(
        n=("Compound_ID", "count"),
        mean_delta=("delta_Consensus", "mean"),
        median_delta=("delta_Consensus", "median"),
        mae=("abs_delta_Consensus", "mean"),
        overestimated_percent=("Overestimated", lambda x: float(x.mean() * 100)),
        severe_n=("Severe_error", "sum"),
    )
    .reindex(group_order)
)

with open(STATS_TXT, "w", encoding="utf-8") as f:
    f.write("Figure 3 consensus error distribution statistics\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"n: {len(errors)}\n")
    f.write(f"Mean delta_Consensus: {mean_error:.6f}\n")
    f.write(f"Median delta_Consensus: {median_error:.6f}\n")
    f.write(f"SD delta_Consensus: {std_error:.6f}\n")
    f.write(f"Overestimated compounds (% delta < 0): {overestimated_percent:.3f}\n")
    f.write(f"|delta| > 1.0 (%): {abs_gt_1_percent:.3f}\n")
    f.write(f"|delta| >= 2.0 (%): {abs_ge_2_percent:.3f}\n")
    f.write(f"Shapiro-Wilk W: {shapiro_w:.6f}\n")
    f.write(f"Shapiro-Wilk p-value: {shapiro_p:.6e}\n\n")
    f.write("Nitrogen-count group summary\n")
    f.write("-" * 72 + "\n")
    f.write(group_summary.to_string())
    f.write("\n\n")
    f.write("Definitions\n")
    f.write("-" * 72 + "\n")
    f.write("delta_Consensus = experimental logP - predicted consensus logP\n")
    f.write("Negative delta_Consensus indicates overestimation of lipophilicity.\n")


# ============================================================
# 5. Plot settings
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 6. Figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(8.5, 4.4),
    gridspec_kw={"width_ratios": [1.18, 0.92]},
)

# Colors
hist_color = "#8DBAD7"
kde_color = "#1F6FB5"
mean_color = "#D62728"
zero_color = "0.35"

group_colors = {
    "N = 0": "#4C78A8",
    "N = 1": "#F2B45A",
    "N = 2–3": "#F28E2B",
    "N ≥ 4": "#8E5A9E",
}

# ------------------------------------------------------------
# Panel A: histogram + KDE
# ------------------------------------------------------------

bins = np.linspace(-5.8, 2.6, 20)

ax1.hist(
    errors,
    bins=bins,
    density=True,
    color=hist_color,
    edgecolor="white",
    linewidth=0.55,
    alpha=0.80,
    label="Histogram",
)

# KDE
x_grid = np.linspace(errors.min() - 0.4, errors.max() + 0.4, 400)
kde = stats.gaussian_kde(errors)
ax1.plot(
    x_grid,
    kde(x_grid),
    color=kde_color,
    linewidth=1.7,
    label="KDE",
)

# Mean and zero lines
ax1.axvline(
    0.0,
    color=zero_color,
    linestyle="--",
    linewidth=1.0,
    alpha=0.85,
)

ax1.axvline(
    mean_error,
    color=mean_color,
    linestyle="-",
    linewidth=1.3,
    alpha=0.95,
)

# Annotation box
annotation = (
    f"Mean = {mean_error:.3f}\n"
    f"Median = {median_error:.3f}\n"
    f"Overestimated = {overestimated_percent:.1f}%\n"
    f"Shapiro–Wilk W = {shapiro_w:.3f}, p = {shapiro_p:.3f}"
)

ax1.text(
    0.03,
    0.96,
    annotation,
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=8.6,
    bbox=dict(
        boxstyle="round,pad=0.32",
        facecolor="white",
        edgecolor="0.55",
        linewidth=0.6,
        alpha=0.95,
    ),
)

ax1.set_xlabel(r"Consensus error, $\Delta$logP")
ax1.set_ylabel("Density")

ax1.set_xlim(-5.8, 2.6)
ax1.set_ylim(bottom=0)

ax1.legend(
    loc="upper right",
    fontsize=8.5,
    frameon=False,
    handlelength=1.4,
)

ax1.grid(axis="y", linewidth=0.45, alpha=0.30)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.text(-0.12, 1.03, "A", transform=ax1.transAxes, fontsize=12.5, fontweight="bold")


# ------------------------------------------------------------
# Panel B: boxplot by nitrogen-count group
# ------------------------------------------------------------

box_data = [
    work.loc[work["N_group"] == g, "delta_Consensus"].values
    for g in group_order
]

positions = np.arange(1, len(group_order) + 1)

bp = ax2.boxplot(
    box_data,
    positions=positions,
    widths=0.55,
    patch_artist=True,
    showfliers=True,
    medianprops=dict(color="black", linewidth=1.0),
    boxprops=dict(color="black", linewidth=0.9),
    whiskerprops=dict(color="black", linewidth=0.8),
    capprops=dict(color="black", linewidth=0.8),
    flierprops=dict(
        marker="o",
        markerfacecolor="0.65",
        markeredgecolor="0.65",
        markersize=3,
        alpha=0.65,
    ),
)

for patch, g in zip(bp["boxes"], group_order):
    patch.set_facecolor(group_colors[g])
    patch.set_alpha(0.82)

# Deterministic jitter overlay
rng = np.random.default_rng(20260609)
for idx, g in enumerate(group_order, start=1):
    vals = work.loc[work["N_group"] == g, "delta_Consensus"].values
    x_jitter = idx + rng.normal(0, 0.045, size=len(vals))
    ax2.scatter(
        x_jitter,
        vals,
        s=17,
        color=group_colors[g],
        alpha=0.60,
        edgecolors="none",
        zorder=3,
    )

# Zero line
ax2.axhline(
    0.0,
    linestyle="--",
    color=zero_color,
    linewidth=1.0,
    alpha=0.85,
)

# X labels with n
x_labels = [
    f"{g}\n(n = {int(group_summary.loc[g, 'n'])})"
    for g in group_order
]

ax2.set_xticks(positions)
ax2.set_xticklabels(x_labels)

ax2.set_ylabel(r"Consensus error, $\Delta$logP")
ax2.set_xlabel("Nitrogen-count group")

y_min = min(errors.min() - 0.35, -5.5)
y_max = max(errors.max() + 0.35, 2.5)
ax2.set_ylim(y_min, y_max)

ax2.grid(axis="y", linewidth=0.45, alpha=0.30)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.text(-0.12, 1.03, "B", transform=ax2.transAxes, fontsize=12.5, fontweight="bold")


plt.tight_layout(w_pad=1.5)


# ============================================================
# 7. Save
# ============================================================

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")
fig.savefig(OUT_TIF, dpi=600, bbox_inches="tight")

plt.close(fig)

print("Figure 3 generated successfully.")
print(f"PNG:        {OUT_PNG}")
print(f"PDF:        {OUT_PDF}")
print(f"SVG:        {OUT_SVG}")
print(f"TIF:        {OUT_TIF}")
print(f"Source CSV: {SOURCE_CSV}")
print(f"Stats TXT:  {STATS_TXT}")
print("\nKey statistics:")
print(f"Mean delta_Consensus: {mean_error:.3f}")
print(f"Median delta_Consensus: {median_error:.3f}")
print(f"Overestimated: {overestimated_percent:.1f}%")
print(f"Shapiro-Wilk W={shapiro_w:.3f}, p={shapiro_p:.3f}")