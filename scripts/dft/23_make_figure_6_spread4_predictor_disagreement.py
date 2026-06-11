from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# Figure 6
# Spread4 predictor disagreement as a warning signal
# Two-panel final version for manuscript
# ============================================================

PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

OUT_FIG_DIR = PROJECT_DIR / "figures" / "main"
OUT_SRC_DIR = PROJECT_DIR / "figures" / "source_data"
OUT_DATA_DIR = PROJECT_DIR / "data" / "processed"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SRC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    norm = {c.lower().replace(" ", "").replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("-", "_")
        if key in norm:
            return norm[key]
    raise KeyError(
        f"None of the candidate columns were found: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

compound_col = find_column(df, ["Compound_ID", "compound_id", "ID"])
logp_col = find_column(df, ["logP_exp", "LogP_exp", "experimental_logP", "Exp_logP"])
xlogp3_col = find_column(df, ["XLOGP3", "xlogp3"])
wlogp_col = find_column(df, ["WLOGP", "wlogp"])
mlogp_col = find_column(df, ["MLOGP", "mlogp"])
silicos_col = find_column(df, ["Silicos-IT", "Silicos_IT", "SilicosIT", "silicos_it"])
consensus_col = find_column(df, ["Consensus", "consensus", "Consensus_logP", "consensus_logP"])

try:
    delta_consensus_col = find_column(
        df,
        ["delta_Consensus", "Delta_Consensus", "delta_consensus", "Consensus_delta"]
    )
    use_stored_delta = True
except KeyError:
    use_stored_delta = False


# ------------------------------------------------------------
# Prepare working dataframe
# ------------------------------------------------------------
work = df[
    [
        compound_col,
        logp_col,
        xlogp3_col,
        wlogp_col,
        mlogp_col,
        silicos_col,
        consensus_col,
    ]
].copy()

work.columns = [
    "Compound_ID",
    "logP_exp",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos_IT",
    "Consensus",
]

fragment_cols = ["XLOGP3", "WLOGP", "MLOGP", "Silicos_IT"]

work["Spread4"] = work[fragment_cols].max(axis=1) - work[fragment_cols].min(axis=1)

if use_stored_delta:
    work["delta_Consensus"] = df[delta_consensus_col].astype(float)
else:
    work["delta_Consensus"] = work["logP_exp"] - work["Consensus"]

work["abs_delta_Consensus"] = work["delta_Consensus"].abs()
work["Severe_error"] = work["abs_delta_Consensus"] >= 2.0

# Deterministic jitter
rng = np.random.default_rng(20260609)

work["Spread4_jittered"] = work["Spread4"] + rng.normal(0, 0.03, size=len(work))
work["abs_delta_jittered"] = work["abs_delta_Consensus"] + rng.normal(0, 0.03, size=len(work))
work["abs_delta_jittered"] = work["abs_delta_jittered"].clip(lower=0)

# Create severe / non-severe subsets after jitter columns are added
severe = work[work["Severe_error"]].copy()
non_severe = work[~work["Severe_error"]].copy()

non_severe["x_jitter"] = 1 + rng.normal(0, 0.04, size=len(non_severe))
severe["x_jitter"] = 2 + rng.normal(0, 0.04, size=len(severe))


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
pearson_r, pearson_p = stats.pearsonr(work["Spread4"], work["abs_delta_Consensus"])
spearman_rho, spearman_p = stats.spearmanr(work["Spread4"], work["abs_delta_Consensus"])

mw = stats.mannwhitneyu(
    non_severe["Spread4"],
    severe["Spread4"],
    alternative="two-sided"
)

slope, intercept, r_value, p_value, stderr = stats.linregress(
    work["Spread4"], work["abs_delta_Consensus"]
)

x_line = np.linspace(
    max(0, work["Spread4"].min() - 0.1),
    work["Spread4"].max() + 0.2,
    200
)
y_line = intercept + slope * x_line


# ------------------------------------------------------------
# Save source data / stats
# ------------------------------------------------------------
source_file = OUT_SRC_DIR / "Figure_6_predictor_disagreement_Spread4_source_data.csv"
stats_file = OUT_DATA_DIR / "Dataset_S11_predictor_disagreement_statistics.txt"
processed_file = OUT_DATA_DIR / "Dataset_S10_predictor_disagreement_Spread4.csv"

save_cols = [
    "Compound_ID",
    "logP_exp",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "Silicos_IT",
    "Consensus",
    "Spread4",
    "delta_Consensus",
    "abs_delta_Consensus",
    "Severe_error",
]
work[save_cols].to_csv(source_file, index=False, encoding="utf-8-sig")
work[save_cols].to_csv(processed_file, index=False, encoding="utf-8-sig")

with open(stats_file, "w", encoding="utf-8") as f:
    f.write("Dataset_S11 predictor-disagreement statistics\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Total compounds: {len(work)}\n")
    f.write(f"Non-severe compounds (|ΔlogP| < 2.0): {len(non_severe)}\n")
    f.write(f"Severe-error compounds (|ΔlogP| ≥ 2.0): {len(severe)}\n\n")

    f.write("Correlation between Spread4 and |ΔlogP|\n")
    f.write("-" * 72 + "\n")
    f.write(f"Pearson r: {pearson_r:.6f}\n")
    f.write(f"Pearson p-value: {pearson_p:.6e}\n")
    f.write(f"Spearman rho: {spearman_rho:.6f}\n")
    f.write(f"Spearman p-value: {spearman_p:.6e}\n\n")

    f.write("Group comparison of Spread4\n")
    f.write("-" * 72 + "\n")
    f.write(f"Mean Spread4, non-severe: {non_severe['Spread4'].mean():.6f}\n")
    f.write(f"Median Spread4, non-severe: {non_severe['Spread4'].median():.6f}\n")
    f.write(f"Mean Spread4, severe: {severe['Spread4'].mean():.6f}\n")
    f.write(f"Median Spread4, severe: {severe['Spread4'].median():.6f}\n")
    f.write(f"Mann-Whitney U statistic: {mw.statistic:.6f}\n")
    f.write(f"Mann-Whitney p-value: {mw.pvalue:.6e}\n\n")

    f.write("Linear fit for visual guide\n")
    f.write("-" * 72 + "\n")
    f.write(f"Slope: {slope:.6f}\n")
    f.write(f"Intercept: {intercept:.6f}\n")
    f.write(f"Linregress r: {r_value:.6f}\n")
    f.write(f"Linregress p-value: {p_value:.6e}\n")
    f.write(f"Standard error of slope: {stderr:.6f}\n")


# ------------------------------------------------------------
# Figure style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(8.2, 4.3),
    gridspec_kw={"width_ratios": [1.18, 0.92]}
)

# Colors
color_non_severe = "#5DA5DA"
color_severe = "#F28E2B"
line_color = "#2C7FB8"


# ------------------------------------------------------------
# Panel A: Scatter
# ------------------------------------------------------------
ax1.scatter(
    non_severe["Spread4_jittered"],
    non_severe["abs_delta_jittered"],
    s=24,
    alpha=0.85,
    edgecolors="none",
    label=r"|ΔlogP| < 2.0",
    zorder=3
)

ax1.scatter(
    severe["Spread4_jittered"],
    severe["abs_delta_jittered"],
    s=32,
    alpha=0.90,
    edgecolors="none",
    label=r"|ΔlogP| ≥ 2.0",
    zorder=4
)

ax1.plot(
    x_line,
    y_line,
    linewidth=1.2,
    color=line_color,
    zorder=2
)

ax1.axhline(
    2.0,
    linestyle="--",
    linewidth=1.0,
    color=color_non_severe,
    zorder=1
)

# Label the main outlier
if "CMR_GOLD_058" in work["Compound_ID"].values:
    row_058 = work.loc[work["Compound_ID"] == "CMR_GOLD_058"].iloc[0]
    ax1.text(
        row_058["Spread4"] + 0.12,
        row_058["abs_delta_Consensus"] + 0.12,
        "CMR_GOLD_058",
        fontsize=8.3,
        ha="left",
        va="bottom"
    )

ax1.text(
    0.05, 0.965,
    f"Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_rho:.3f}",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=9.2
)

ax1.set_xlabel("Predictor disagreement, Spread4")
ax1.set_ylabel(r"Absolute consensus error, |ΔlogP|")
ax1.set_xlim(0.0, max(4.0, work["Spread4"].max() + 0.15))
ax1.set_ylim(-0.1, max(5.5, work["abs_delta_Consensus"].max() + 0.35))
ax1.grid(axis="y", linewidth=0.5, alpha=0.35)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(
    loc="lower right",
    frameon=False,
    fontsize=8.7,
    handletextpad=0.4,
    borderpad=0.2
)
ax1.text(-0.12, 1.03, "A", transform=ax1.transAxes, fontsize=12, fontweight="bold")


# ------------------------------------------------------------
# Panel B: Boxplot + jitter
# ------------------------------------------------------------
box_data = [non_severe["Spread4"].values, severe["Spread4"].values]

bp = ax2.boxplot(
    box_data,
    positions=[1, 2],
    widths=0.45,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.0),
    boxprops=dict(color="black", linewidth=0.9),
    whiskerprops=dict(color="black", linewidth=0.8),
    capprops=dict(color="black", linewidth=0.8)
)

bp["boxes"][0].set_facecolor("#B8C7D9")
bp["boxes"][0].set_alpha(0.9)
bp["boxes"][1].set_facecolor("#E7D0B5")
bp["boxes"][1].set_alpha(0.95)

ax2.scatter(
    non_severe["x_jitter"],
    non_severe["Spread4"],
    s=23,
    alpha=0.85,
    edgecolors="none",
    zorder=3
)

ax2.scatter(
    severe["x_jitter"],
    severe["Spread4"],
    s=25,
    alpha=0.90,
    edgecolors="none",
    zorder=4
)

# significance bar
y_top = max(work["Spread4"].max(), severe["Spread4"].max(), non_severe["Spread4"].max())
bar_y = y_top + 0.10
bar_h = 0.06
ax2.plot([1, 1, 2, 2], [bar_y, bar_y + bar_h, bar_y + bar_h, bar_y], color="black", linewidth=0.9)
ax2.text(
    1.5,
    bar_y + bar_h + 0.06,
    r"Mann–Whitney $p$ = 4.82 × 10$^{-4}$",
    ha="center",
    va="bottom",
    fontsize=9.2
)

ax2.set_ylabel("Spread4 predictor disagreement")
ax2.set_xticks([1, 2])
ax2.set_xticklabels([
    f"|ΔlogP| < 2.0\n(n = {len(non_severe)})",
    f"|ΔlogP| ≥ 2.0\n(n = {len(severe)})"
])
ax2.set_ylim(0, max(4.2, y_top + 0.38))
ax2.grid(axis="y", linewidth=0.5, alpha=0.35)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.text(-0.10, 1.03, "B", transform=ax2.transAxes, fontsize=12, fontweight="bold")


plt.tight_layout(w_pad=1.3)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
png_path = OUT_FIG_DIR / "Figure_6_predictor_disagreement_Spread4.png"
pdf_path = OUT_FIG_DIR / "Figure_6_predictor_disagreement_Spread4.pdf"
svg_path = OUT_FIG_DIR / "Figure_6_predictor_disagreement_Spread4.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print("Figure 6 generated successfully.")
print(f"Source data : {source_file}")
print(f"Processed   : {processed_file}")
print(f"Statistics  : {stats_file}")
print(f"PNG         : {png_path}")
print(f"PDF         : {pdf_path}")
print(f"SVG         : {svg_path}")