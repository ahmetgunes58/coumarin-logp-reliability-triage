# -*- coding: utf-8 -*-
"""
Figure 1
Performance of SwissADME-associated logP predictors across the curated coumarin dataset.
Final main-text re-export / polish version.

Run:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python scripts\28_make_figure_1_predictor_performance.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUT_PNG = OUT_FIG_DIR / "Figure_1_predictor_performance.png"
OUT_PDF = OUT_FIG_DIR / "Figure_1_predictor_performance.pdf"
OUT_SVG = OUT_FIG_DIR / "Figure_1_predictor_performance.svg"
OUT_TIF = OUT_FIG_DIR / "Figure_1_predictor_performance.tif"

SOURCE_CSV = OUT_SRC_DIR / "Figure_1_predictor_performance_source_data.csv"
METRICS_CSV = OUT_SRC_DIR / "Figure_1_predictor_performance_metrics.csv"
METRICS_TXT = OUT_DATA_DIR / "Dataset_Figure_1_predictor_performance_metrics.txt"


# ============================================================
# 2. Helpers
# ============================================================

def normalise_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    norm_map = {normalise_col(c): c for c in df.columns}

    for candidate in candidates:
        key = normalise_col(candidate)
        if key in norm_map:
            return norm_map[key]

    raise KeyError(
        f"None of the candidate columns were found: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def n_group_from_count(n: int) -> str:
    n = int(n)
    if n == 0:
        return "N = 0"
    if n == 1:
        return "N = 1"
    if n in (2, 3):
        return "N = 2–3"
    return "N ≥ 4"


def calculate_metrics(y_exp: np.ndarray, y_pred: np.ndarray) -> dict:
    delta = y_exp - y_pred
    bias = float(np.mean(delta))
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta ** 2)))

    ss_res = float(np.sum((y_exp - y_pred) ** 2))
    ss_tot = float(np.sum((y_exp - np.mean(y_exp)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "Bias": bias,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# 3. Load data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

compound_col = find_column(df, ["Compound_ID", "compound_id", "ID"])
logp_col = find_column(df, ["logP_exp", "LogP_exp", "experimental_logP", "Exp_logP"])
n_count_col = find_column(df, ["N_count", "n_count", "Nitrogen_count"])

predictor_columns = {
    "iLOGP": find_column(df, ["iLOGP", "ilogp"]),
    "MLOGP": find_column(df, ["MLOGP", "mlogp"]),
    "Consensus": find_column(df, ["Consensus", "consensus", "Consensus_logP", "consensus_logP"]),
    "WLOGP": find_column(df, ["WLOGP", "wlogp"]),
    "XLOGP3": find_column(df, ["XLOGP3", "xlogp3"]),
    "Silicos-IT": find_column(df, ["Silicos-IT", "Silicos_IT", "SilicosIT", "silicos_it"]),
}

plot_df = df[[compound_col, logp_col, n_count_col] + list(predictor_columns.values())].copy()

plot_df = plot_df.rename(
    columns={
        compound_col: "Compound_ID",
        logp_col: "logP_exp",
        n_count_col: "N_count",
        **{v: k for k, v in predictor_columns.items()},
    }
)

numeric_cols = ["logP_exp", "N_count"] + list(predictor_columns.keys())
for col in numeric_cols:
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

plot_df = plot_df.dropna(subset=["logP_exp", "N_count"]).copy()
plot_df["N_count"] = plot_df["N_count"].astype(int)
plot_df["N_group"] = plot_df["N_count"].apply(n_group_from_count)

predictor_order = ["iLOGP", "MLOGP", "Consensus", "WLOGP", "XLOGP3", "Silicos-IT"]
n_group_order = ["N = 0", "N = 1", "N = 2–3", "N ≥ 4"]

# Keep only rows that have all predictor values for this figure
plot_df = plot_df.dropna(subset=["logP_exp"] + predictor_order).copy()


# ============================================================
# 4. Metrics and source data
# ============================================================

source_rows = []
metric_rows = []

for predictor in predictor_order:
    y_exp = plot_df["logP_exp"].to_numpy(dtype=float)
    y_pred = plot_df[predictor].to_numpy(dtype=float)

    metrics = calculate_metrics(y_exp, y_pred)

    metric_rows.append(
        {
            "Predictor": predictor,
            "n": len(plot_df),
            "Bias": metrics["Bias"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        }
    )

    tmp = plot_df[["Compound_ID", "logP_exp", "N_count", "N_group", predictor]].copy()
    tmp = tmp.rename(columns={predictor: "logP_pred"})
    tmp["Predictor"] = predictor
    tmp["Delta_logP"] = tmp["logP_exp"] - tmp["logP_pred"]
    source_rows.append(tmp)

source_df = pd.concat(source_rows, ignore_index=True)
metrics_df = pd.DataFrame(metric_rows)

source_df.to_csv(SOURCE_CSV, index=False, encoding="utf-8-sig")
metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

with open(METRICS_TXT, "w", encoding="utf-8") as f:
    f.write("Figure 1 predictor-performance metrics\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"n: {len(plot_df)}\n\n")
    f.write("Definitions\n")
    f.write("-" * 72 + "\n")
    f.write("Bias = logP_exp - logP_pred\n")
    f.write("Negative bias indicates overestimation of lipophilicity.\n")
    f.write("R2 = 1 - SSE/SST using experimental logP as reference.\n\n")
    f.write(metrics_df.to_string(index=False))


# ============================================================
# 5. Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 10.2,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

group_colors = {
    "N = 0": "#4C78A8",
    "N = 1": "#F2B45A",
    "N = 2–3": "#F28E2B",
    "N ≥ 4": "#8E5A9E",
}

predictor_title_colors = {
    "iLOGP": "#2A6FBB",
    "MLOGP": "#7B3FB2",
    "Consensus": "#1B8E3E",
    "WLOGP": "#D6278B",
    "XLOGP3": "#2CA02C",
    "Silicos-IT": "#E56B00",
}


# ============================================================
# 6. Figure
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.15), sharex=True, sharey=True)
axes = axes.flatten()

# Common axis limits
all_values = np.concatenate(
    [
        plot_df["logP_exp"].to_numpy(dtype=float),
        *[plot_df[p].to_numpy(dtype=float) for p in predictor_order],
    ]
)

axis_min = np.floor(np.nanmin(all_values) - 0.5)
axis_max = np.ceil(np.nanmax(all_values) + 0.5)

# keep manuscript-friendly limits
axis_min = min(axis_min, -1.5)
axis_max = max(axis_max, 6.5)

x_line = np.linspace(axis_min, axis_max, 300)

for idx, (ax, predictor) in enumerate(zip(axes, predictor_order)):
    # near-diagonal visual reference band: ±1 log unit
    ax.fill_between(
        x_line,
        x_line - 1.0,
        x_line + 1.0,
        color="0.88",
        alpha=0.55,
        zorder=0,
    )

    # ideal line
    ax.plot(
        x_line,
        x_line,
        linestyle="--",
        color="0.45",
        linewidth=1.0,
        zorder=1,
    )

    # scatter by N group
    for group in n_group_order:
        sub = plot_df[plot_df["N_group"] == group]
        ax.scatter(
            sub["logP_exp"],
            sub[predictor],
            s=21,
            color=group_colors[group],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.25,
            label=group,
            zorder=3,
        )

    # metrics box
    row = metrics_df[metrics_df["Predictor"] == predictor].iloc[0]
    metrics_text = (
        f"RMSE = {row['RMSE']:.2f}\n"
        f"Bias = {row['Bias']:.2f}\n"
        f"R² = {row['R2']:.2f}"
    )

    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor="0.60",
            linewidth=0.55,
            alpha=0.94,
        ),
        zorder=5,
    )

    # panel label
    panel_letter = chr(ord("A") + idx)
    ax.text(
        -0.16,
        1.06,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
    )

    # predictor title inside/above panel
    ax.set_title(
        predictor,
        fontsize=10.3,
        fontweight="bold",
        color=predictor_title_colors[predictor],
        pad=5,
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if idx % 3 == 0:
        ax.set_ylabel("Predicted logP")

    if idx >= 3:
        ax.set_xlabel("Experimental logP")


# Legend in the last panel area but outside plotting points
handles, labels = axes[-1].get_legend_handles_labels()
unique = dict(zip(labels, handles))

fig.legend(
    unique.values(),
    unique.keys(),
    title="N group",
    loc="lower right",
    bbox_to_anchor=(0.985, 0.105),
    frameon=False,
    fontsize=8.2,
    title_fontsize=8.8,
    handletextpad=0.35,
    labelspacing=0.30,
)

plt.subplots_adjust(
    left=0.075,
    right=0.965,
    top=0.935,
    bottom=0.105,
    wspace=0.32,
    hspace=0.42,
)


# ============================================================
# 7. Save
# ============================================================

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")
fig.savefig(OUT_TIF, dpi=600, bbox_inches="tight")

plt.close(fig)

print("Figure 1 generated successfully.")
print(f"PNG:         {OUT_PNG}")
print(f"PDF:         {OUT_PDF}")
print(f"SVG:         {OUT_SVG}")
print(f"TIF:         {OUT_TIF}")
print(f"Source CSV:  {SOURCE_CSV}")
print(f"Metrics CSV: {METRICS_CSV}")
print(f"Metrics TXT: {METRICS_TXT}")

print("\nMetrics:")
print(metrics_df.to_string(index=False))