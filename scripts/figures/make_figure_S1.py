# -*- coding: utf-8 -*-
"""
Figure S1. Structural-class distribution across the curated coumarin dataset.

Input:
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/data/processed/Dataset_S4_structural_class_summary.csv

Outputs:
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/figures/supporting/Figure_S1_structural_class_distribution.png
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/figures/supporting/Figure_S1_structural_class_distribution.tiff
    C:/Users/Ahmet Gunes/YandexDisk/Makaleler/4.1-/coumarin-logp/coumarin-logp/figures/supporting/Figure_S1_structural_class_distribution.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")

INPUT_CSV = ROOT / "data" / "processed" / "Dataset_S4_structural_class_summary.csv"

OUTPUT_DIR = ROOT / "figures" / "supporting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "Figure_S1_structural_class_distribution.png"
OUT_TIFF = OUTPUT_DIR / "Figure_S1_structural_class_distribution.tiff"
OUT_PDF = OUTPUT_DIR / "Figure_S1_structural_class_distribution.pdf"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_columns = ["Structural_Class_SI", "n"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Eksik sütun(lar): {missing_columns}")

df["n"] = pd.to_numeric(df["n"], errors="coerce")

if df["n"].isna().any():
    raise ValueError("'n' sütununda sayısal olmayan değer(ler) var.")

total_n = int(df["n"].sum())
if total_n != 95:
    raise ValueError(f"Toplam n = 95 olmalıydı, fakat {total_n} bulundu.")


# ============================================================
# Preserve SI order
# Horizontal bar plot için ters çeviriyoruz
# ============================================================

plot_df = df.iloc[::-1].copy()


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

fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=600)

bars = ax.barh(
    plot_df["Structural_Class_SI"],
    plot_df["n"],
    height=0.62,
    color="#4C78A8",
    edgecolor="black",
    linewidth=0.4
)

# Bar sonlarına n değerlerini yaz
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.45,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width)}",
        va="center",
        ha="left",
        fontsize=8.5
    )

# Axis labels
ax.set_xlabel("Number of compounds", fontsize=9.5)
ax.set_ylabel("")

# X limit: fazla boşluğu azalt
max_n = int(plot_df["n"].max())
ax.set_xlim(0, max_n + 5)

# Grid
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
ax.set_axisbelow(True)

# Spine cleanup
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Tick style
ax.tick_params(axis="y", labelsize=8.8)
ax.tick_params(axis="x", labelsize=8.8)

fig.tight_layout()

# Save
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

plt.close(fig)

print("Figure S1 başarıyla oluşturuldu.")
print(f"Girdi dosyası : {INPUT_CSV}")
print(f"PNG çıktı     : {OUT_PNG}")
print(f"TIFF çıktı    : {OUT_TIFF}")
print(f"PDF çıktı     : {OUT_PDF}")
print(f"Toplam n      : {total_n}")