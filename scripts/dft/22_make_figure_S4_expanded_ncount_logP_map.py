from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# ============================================================
# Figure S4
# Expanded nitrogen-count × logP risk map
# with selected ten-compound DFT/MEP diagnostic-panel labels
# ============================================================

PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

OUT_FIG_DIR = PROJECT_DIR / "figures" / "supplementary"
OUT_SRC_DIR = PROJECT_DIR / "figures" / "source_data"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SRC_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
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


def y_position_from_group(group: str) -> int:
    mapping = {
        "N = 0": 0,
        "N = 1": 1,
        "N = 2–3": 2,
        "N ≥ 4": 3,
    }
    return mapping[group]


def short_id(compound_id: str) -> str:
    # CMR_GOLD_058 -> 058
    return compound_id.split("_")[-1]


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

compound_col = find_column(df, ["Compound_ID", "compound_id", "ID"])
logp_col = find_column(df, ["logP_exp", "LogP_exp", "experimental_logP", "Exp_logP"])
n_count_col = find_column(df, ["N_count", "n_count", "Nitrogen_count"])
delta_col = find_column(
    df,
    ["delta_Consensus", "Delta_Consensus", "delta_consensus", "Consensus_delta"]
)

plot_df = df[[compound_col, logp_col, n_count_col, delta_col]].copy()
plot_df.columns = ["Compound_ID", "logP_exp", "N_count", "delta_Consensus"]

plot_df["N_count"] = plot_df["N_count"].astype(int)
plot_df["N_group"] = plot_df["N_count"].apply(n_group_from_count)
plot_df["N_group_y"] = plot_df["N_group"].apply(y_position_from_group)
plot_df["abs_delta_Consensus"] = plot_df["delta_Consensus"].abs()
plot_df["Severe_error"] = plot_df["abs_delta_Consensus"] >= 2.0

# Deterministic jitter for reproducible plotting
rng = np.random.default_rng(20260609)
plot_df["x_jittered"] = plot_df["logP_exp"] + rng.normal(0, 0.035, size=len(plot_df))
plot_df["y_jittered"] = plot_df["N_group_y"] + rng.normal(0, 0.045, size=len(plot_df))


# ------------------------------------------------------------
# Ten-compound DFT/MEP diagnostic panel
# ------------------------------------------------------------
dft_panel = [
    "CMR_GOLD_055",
    "CMR_GOLD_043",
    "CMR_GOLD_044",
    "CMR_GOLD_029",
    "CMR_GOLD_058",
    "CMR_GOLD_079",
    "CMR_GOLD_016",
    "CMR_GOLD_090",
    "CMR_GOLD_020",
    "CMR_GOLD_092",
]

plot_df["DFT_panel"] = plot_df["Compound_ID"].isin(dft_panel)
plot_df["DFT_short_label"] = np.where(
    plot_df["DFT_panel"],
    plot_df["Compound_ID"].apply(short_id),
    "",
)


# ------------------------------------------------------------
# Save source data
# ------------------------------------------------------------
source_file = OUT_SRC_DIR / "Figure_S4_expanded_ncount_logP_map_source_data.csv"

plot_df[
    [
        "Compound_ID",
        "DFT_short_label",
        "logP_exp",
        "N_count",
        "N_group",
        "N_group_y",
        "x_jittered",
        "y_jittered",
        "delta_Consensus",
        "abs_delta_Consensus",
        "Severe_error",
        "DFT_panel",
    ]
].to_csv(source_file, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(8.8, 5.1))


# ------------------------------------------------------------
# Background regions
# ------------------------------------------------------------
# Vertical logP bins
ax.axvspan(0.0, 1.5, color="0.92", alpha=0.75, zorder=0)
ax.axvspan(1.5, 3.0, color="0.97", alpha=0.75, zorder=0)
ax.axvspan(3.0, 6.2, color="0.92", alpha=0.45, zorder=0)

# Principal high-risk region: polar N = 1–3 domain
ax.add_patch(
    Rectangle(
        (0.0, 1.5),
        1.5,
        1.0,
        facecolor="0.84",
        edgecolor="none",
        alpha=0.65,
        zorder=0,
    )
)

# Vertical logP boundaries
for x in [1.5, 3.0]:
    ax.axvline(
        x,
        linestyle="--",
        linewidth=0.9,
        color="black",
        alpha=0.55,
        zorder=1,
    )

# Horizontal N-group separators
for y in [0.5, 1.5, 2.5]:
    ax.axhline(
        y,
        linestyle="-",
        linewidth=0.5,
        color="0.82",
        zorder=1,
    )


# ------------------------------------------------------------
# Scatter points
# ------------------------------------------------------------
non_severe = plot_df[~plot_df["Severe_error"]]
severe = plot_df[plot_df["Severe_error"]]

# All compounds, colour encodes severe-error category
ax.scatter(
    non_severe["x_jittered"],
    non_severe["y_jittered"],
    s=25,
    alpha=0.65,
    edgecolors="none",
    label=r"|ΔlogP| < 2.0",
    zorder=3,
)

ax.scatter(
    severe["x_jittered"],
    severe["y_jittered"],
    s=34,
    alpha=0.82,
    edgecolors="black",
    linewidths=0.35,
    label=r"|ΔlogP| ≥ 2.0",
    zorder=4,
)

# DFT-panel overlay: hollow diamond marker
dft = plot_df[plot_df["DFT_panel"]]

ax.scatter(
    dft["x_jittered"],
    dft["y_jittered"],
    s=78,
    marker="D",
    facecolors="none",
    edgecolors="black",
    linewidths=1.05,
    zorder=7,
)


# ------------------------------------------------------------
# Region labels
# ------------------------------------------------------------
ax.text(
    0.75,
    3.31,
    "polar\nlogP < 1.5",
    ha="center",
    va="bottom",
    fontsize=10,
)

ax.text(
    2.25,
    3.31,
    "intermediate\n1.5 ≤ logP < 3.0",
    ha="center",
    va="bottom",
    fontsize=10,
)

ax.text(
    4.60,
    3.31,
    "higher logP\n≥ 3.0",
    ha="center",
    va="bottom",
    fontsize=10,
)

ax.text(
    0.18,
    2.40,
    "principal\nhigh-risk zone",
    ha="left",
    va="center",
    fontsize=8.8,
    fontweight="bold",
)


# ------------------------------------------------------------
# Short labels for DFT-panel compounds
# ------------------------------------------------------------
label_offsets = {
    "CMR_GOLD_055": (0.06, 0.12),
    "CMR_GOLD_043": (0.06, -0.16),
    "CMR_GOLD_044": (0.06, 0.13),
    "CMR_GOLD_029": (0.06, -0.17),
    "CMR_GOLD_058": (0.06, 0.15),
    "CMR_GOLD_079": (0.06, 0.14),
    "CMR_GOLD_016": (0.06, -0.14),
    "CMR_GOLD_090": (0.06, 0.11),
    "CMR_GOLD_020": (0.06, -0.15),
    "CMR_GOLD_092": (0.06, 0.13),
}

for compound_id, (dx, dy) in label_offsets.items():
    row = plot_df.loc[plot_df["Compound_ID"] == compound_id]

    if row.empty:
        print(f"Warning: {compound_id} not found in Dataset_S1.")
        continue

    x = float(row["x_jittered"].iloc[0])
    y = float(row["y_jittered"].iloc[0])
    label = str(row["DFT_short_label"].iloc[0])

    ax.text(
        x + dx,
        y + dy,
        label,
        fontsize=7.8,
        ha="center",
        va="center",
        zorder=8,
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor="white",
            edgecolor="0.35",
            linewidth=0.45,
            alpha=0.92,
        ),
    )


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------
ax.set_xlabel("Experimental logP")
ax.set_ylabel("Nitrogen-count group")

ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["N = 0", "N = 1", "N = 2–3", "N ≥ 4"])

ax.set_xlim(-0.3, 6.2)
ax.set_ylim(-0.45, 3.60)

ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# Custom legend outside plotting area
# ------------------------------------------------------------
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=6,
        markerfacecolor="C0",
        markeredgecolor="none",
        alpha=0.65,
        label=r"|ΔlogP| < 2.0",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=6.5,
        markerfacecolor="C1",
        markeredgecolor="black",
        markeredgewidth=0.35,
        alpha=0.85,
        label=r"|ΔlogP| ≥ 2.0",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="None",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.05,
        label="DFT/MEP panel",
    ),
]

ax.legend(
    handles=legend_handles,
    loc="lower left",
    bbox_to_anchor=(1.01, 0.02),
    fontsize=8.8,
    handletextpad=0.5,
    borderpad=0.5,
    labelspacing=0.35,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=0.95,
)


plt.tight_layout()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
png_path = OUT_FIG_DIR / "Figure_S4_expanded_ncount_logP_map.png"
pdf_path = OUT_FIG_DIR / "Figure_S4_expanded_ncount_logP_map.pdf"
svg_path = OUT_FIG_DIR / "Figure_S4_expanded_ncount_logP_map.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

plt.close(fig)

print("Figure S4 generated successfully.")
print(f"Source data : {source_file}")
print(f"PNG         : {png_path}")
print(f"PDF         : {pdf_path}")
print(f"SVG         : {svg_path}")