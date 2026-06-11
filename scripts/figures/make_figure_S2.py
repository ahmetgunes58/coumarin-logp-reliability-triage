# -*- coding: utf-8 -*-
"""
Figure S2. Six-predictor predicted versus experimental logP plots.

Input:
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/data/processed/Dataset_S1_benchmark_dataset.csv

Outputs:
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/figures/supporting/Figure_S2_six_predictor_scatter_plots.png
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/figures/supporting/Figure_S2_six_predictor_scatter_plots.tiff
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/figures/supporting/Figure_S2_six_predictor_scatter_plots.pdf
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/data/processed/Dataset_S18_six_predictor_scatter_source_data.csv
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

FIG_DIR = ROOT / "figures" / "supporting"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = FIG_DIR / "Figure_S2_six_predictor_scatter_plots.png"
OUT_TIFF = FIG_DIR / "Figure_S2_six_predictor_scatter_plots.tiff"
OUT_PDF = FIG_DIR / "Figure_S2_six_predictor_scatter_plots.pdf"

OUT_SOURCE = DATA_DIR / "Dataset_S18_six_predictor_scatter_source_data.csv"


# ============================================================
# Configuration
# ============================================================

predictors = [
    ("iLOGP", "iLOGP"),
    ("XLOGP3", "XLOGP3"),
    ("WLOGP", "WLOGP"),
    ("MLOGP", "MLOGP"),
    ("Silicos_IT", "Silicos-IT"),
    ("Consensus", "Consensus"),
]

required_cols = ["Compound_ID", "logP_exp"] + [p[0] for p in predictors]


# ============================================================
# Helper functions
# ============================================================

def compute_metrics(y_true, y_pred):
    delta = y_true - y_pred
    bias = np.mean(delta)
    mae = np.mean(np.abs(delta))
    rmse = np.sqrt(np.mean(delta ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


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

n_total = len(df)
if n_total != 95:
    print(f"Uyarı: Beklenen n = 95, bulunan n = {n_total}")


# ============================================================
# Save source-data file
# ============================================================

source_cols = ["Compound_ID", "logP_exp"] + [p[0] for p in predictors]
df[source_cols].to_csv(OUT_SOURCE, index=False, encoding="utf-8-sig")


# ============================================================
# Axis limits
# ============================================================

all_values = [df["logP_exp"].values]
for pred_col, _ in predictors:
    all_values.append(df[pred_col].values)

global_min = min(np.nanmin(arr) for arr in all_values)
global_max = max(np.nanmax(arr) for arr in all_values)

padding = 0.4
axis_min = np.floor((global_min - padding) * 2) / 2
axis_max = np.ceil((global_max + padding) * 2) / 2


# ============================================================
# Matplotlib settings
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
# Plot
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2), dpi=600, sharex=True, sharey=True)
axes = axes.flatten()

for ax, (pred_col, panel_title) in zip(axes, predictors):
    x = df["logP_exp"].values
    y = df[pred_col].values

    metrics = compute_metrics(x, y)

    ax.scatter(
        x,
        y,
        s=20,
        alpha=0.80,
        edgecolors="black",
        linewidths=0.3,
        color="#4C78A8",
    )

    # identity line
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        linewidth=1.0,
        color="black"
    )

    ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.30)
    ax.set_axisbelow(True)

    stats_text = (
        f"Bias = {metrics['bias']:.2f}\n"
        f"MAE = {metrics['mae']:.2f}\n"
        f"RMSE = {metrics['rmse']:.2f}\n"
        f"$R^2$ = {metrics['r2']:.2f}"
    )

    ax.text(
        0.04, 0.96, stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.90)
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes[3:]:
    ax.set_xlabel("Experimental logP", fontsize=9.5)

for ax in axes[0::3]:
    ax.set_ylabel("Predicted logP", fontsize=9.5)

fig.tight_layout()

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

plt.close(fig)


print("Figure S2 başarıyla oluşturuldu.")
print(f"Girdi dosyası : {INPUT_CSV}")
print(f"Kaynak veri   : {OUT_SOURCE}")
print(f"PNG çıktı     : {OUT_PNG}")
print(f"TIFF çıktı    : {OUT_TIFF}")
print(f"PDF çıktı     : {OUT_PDF}")
print(f"Kullanılan n  : {n_total}")
print(f"Eksen aralığı : {axis_min} to {axis_max}")