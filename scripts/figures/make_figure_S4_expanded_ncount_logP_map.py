# -*- coding: utf-8 -*-
"""
Figure S4. Expanded nitrogen-count × logP map with selected compound labels.

Input:
    coumarin-logp/data/processed/Dataset_S1_benchmark_dataset.csv

Outputs:
    coumarin-logp/data/processed/Dataset_S20_expanded_ncount_logP_map_source_data.csv

    coumarin-logp/figures/supporting/Figure_S4_expanded_ncount_logP_map.png
    coumarin-logp/figures/supporting/Figure_S4_expanded_ncount_logP_map.tiff
    coumarin-logp/figures/supporting/Figure_S4_expanded_ncount_logP_map.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")

INPUT_CSV = ROOT / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "supporting"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_SOURCE = DATA_DIR / "Dataset_S20_expanded_ncount_logP_map_source_data.csv"

OUT_PNG = FIG_DIR / "Figure_S4_expanded_ncount_logP_map.png"
OUT_TIFF = FIG_DIR / "Figure_S4_expanded_ncount_logP_map.tiff"
OUT_PDF = FIG_DIR / "Figure_S4_expanded_ncount_logP_map.pdf"


# ============================================================
# Required columns
# ============================================================

required_cols = [
    "Compound_ID",
    "logP_exp",
    "N_count",
    "N_group",
    "N_15",
    "logP_15",
    "delta_Consensus",
    "FM",
]

label_compounds = [
    "CMR_GOLD_055",
    "CMR_GOLD_043",
    "CMR_GOLD_079",
    "CMR_GOLD_058",
]

# Fine-tuned label positions
label_offsets = {
    "CMR_GOLD_055": (0.08, 0.16),
    "CMR_GOLD_043": (0.08, -0.18),
    "CMR_GOLD_079": (0.10, 0.16),
    "CMR_GOLD_058": (0.08, 0.16),
}


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Eksik sütun(lar): {missing_cols}")

for col in ["logP_exp", "N_count", "delta_Consensus"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Compound_ID", "logP_exp", "N_count", "delta_Consensus"]).copy()

n_total = len(df)
if n_total != 95:
    print(f"Uyarı: Beklenen n = 95, bulunan n = {n_total}")


# ============================================================
# Derived variables
# ============================================================

df["abs_delta_Consensus"] = df["delta_Consensus"].abs()
df["Severe_Error"] = df["abs_delta_Consensus"] >= 2.0

def n_category(n):
    n = int(n)
    if n == 0:
        return "N = 0"
    if n == 1:
        return "N = 1"
    if n in [2, 3]:
        return "N = 2–3"
    return "N ≥ 4"

df["N_plot_group"] = df["N_count"].apply(n_category)

n_order = ["N = 0", "N = 1", "N = 2–3", "N ≥ 4"]
y_map = {group: i for i, group in enumerate(n_order)}
df["y_base"] = df["N_plot_group"].map(y_map)

if df["y_base"].isna().any():
    raise ValueError("Bazı N_count değerleri N_plot_group kategorisine çevrilemedi.")

# Small deterministic jitter for visibility
rng = np.random.default_rng(42)
df["y_plot"] = df["y_base"] + rng.normal(0, 0.035, size=len(df))

# Principal high-risk zone: N = 1–3 and logP < 1.5
df["Principal_high_risk_zone"] = (
    df["N_count"].between(1, 3) & (df["logP_exp"] < 1.5)
)


# ============================================================
# Save source data
# ============================================================

source_cols = [
    "Compound_ID",
    "logP_exp",
    "N_count",
    "N_group",
    "N_15",
    "logP_15",
    "N_plot_group",
    "delta_Consensus",
    "abs_delta_Consensus",
    "Severe_Error",
    "Principal_high_risk_zone",
    "FM",
]

df[source_cols].to_csv(OUT_SOURCE, index=False, encoding="utf-8-sig")


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

fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=600)

x_min = min(-0.2, df["logP_exp"].min() - 0.35)
x_max = max(6.0, df["logP_exp"].max() + 0.45)

# Background zones
ax.axvspan(x_min, 1.5, color="#FDEBEC", alpha=0.75, zorder=0)
ax.axvspan(1.5, 3.0, color="#F7F7F7", alpha=0.65, zorder=0)
ax.axvspan(3.0, x_max, color="#E8F2FC", alpha=0.55, zorder=0)

# Principal high-risk zone rectangle
risk_rect = Rectangle(
    (x_min, 0.5),         # start x, start y
    1.5 - x_min,          # width
    2.0,                  # height (covers N=1 and N=2–3 rows)
    facecolor="#F6B3B3",
    edgecolor="#B71C1C",
    linewidth=0.9,
    alpha=0.25,
    zorder=1,
)
ax.add_patch(risk_rect)

# Non-severe compounds
non_severe = df[~df["Severe_Error"]]
ax.scatter(
    non_severe["logP_exp"],
    non_severe["y_plot"],
    s=28,
    color="#6E97C4",
    edgecolors="black",
    linewidths=0.25,
    alpha=0.85,
    label="|ΔlogP| < 2.0",
    zorder=3,
)

# Severe-error compounds
severe = df[df["Severe_Error"]]
ax.scatter(
    severe["logP_exp"],
    severe["y_plot"],
    s=44,
    color="#EF6A67",
    edgecolors="black",
    linewidths=0.45,
    alpha=0.95,
    label="|ΔlogP| ≥ 2.0",
    zorder=4,
)

# Selected labels
for compound_id in label_compounds:
    sub = df[df["Compound_ID"].astype(str) == compound_id]
    if sub.empty:
        print(f"Uyarı: {compound_id} bulunamadı.")
        continue

    row = sub.iloc[0]
    dx, dy = label_offsets.get(compound_id, (0.08, 0.12))

    ax.annotate(
        compound_id,
        xy=(row["logP_exp"], row["y_plot"]),
        xytext=(row["logP_exp"] + dx, row["y_plot"] + dy),
        textcoords="data",
        fontsize=7.9,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.6,
            color="black",
            shrinkA=0,
            shrinkB=2,
        ),
        zorder=5,
    )

# Vertical boundaries
for x_cut in [1.5, 3.0]:
    ax.axvline(x_cut, linestyle="--", color="black", linewidth=0.8, alpha=0.55, zorder=2)

# Horizontal separators
for y in [0.5, 1.5, 2.5]:
    ax.axhline(y, linestyle=":", color="black", linewidth=0.55, alpha=0.35, zorder=2)

# Zone labels
ax.text(
    0.65, 3.35,
    "polar\nlogP < 1.5",
    ha="center", va="center",
    fontsize=8.2,
    color="#8E2A23"
)

ax.text(
    2.25, 3.35,
    "intermediate\n1.5 ≤ logP ≤ 3.0",
    ha="center", va="center",
    fontsize=8.2,
    color="#333333"
)

ax.text(
    4.45, 3.35,
    "higher logP\n> 3.0",
    ha="center", va="center",
    fontsize=8.2,
    color="#0D47A1"
)

ax.text(
    0.58, 2.30,
    "principal\nhigh-risk zone",
    ha="center", va="center",
    fontsize=7.7,
    color="#C62828",
    fontweight="bold",
)

# Axes
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.45, 3.55)

ax.set_yticks([y_map[g] for g in n_order])
ax.set_yticklabels(n_order)

ax.set_xlabel("Experimental logP")
ax.set_ylabel("Nitrogen-count group")

ax.legend(
    frameon=False,
    loc="lower right",
    fontsize=8.2,
)

ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.25)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

plt.close(fig)


# ============================================================
# Console output
# ============================================================

print("Figure S4 başarıyla oluşturuldu.")
print(f"Girdi dosyası : {INPUT_CSV}")
print(f"Kaynak veri   : {OUT_SOURCE}")
print(f"PNG çıktı     : {OUT_PNG}")
print(f"TIFF çıktı    : {OUT_TIFF}")
print(f"PDF çıktı     : {OUT_PDF}")
print(f"Kullanılan n  : {n_total}")
print(f"Severe-error n: {int(df['Severe_Error'].sum())}")
print(f"High-risk zone n: {int(df['Principal_high_risk_zone'].sum())}")