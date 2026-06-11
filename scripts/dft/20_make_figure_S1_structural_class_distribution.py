from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Figure S1
# Structural-class distribution across the curated coumarin dataset
# ============================================================

PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

OUT_FIG_DIR = PROJECT_DIR / "figures" / "supplementary"
OUT_SRC_DIR = PROJECT_DIR / "figures" / "source_data"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SRC_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Final source data derived from SI Table S7
# Use short label in figure for readability; keep full terminology in caption.
# ------------------------------------------------------------
data = [
    ("Oxadiazole", 5),
    ("Furocoumarin", 4),
    ("Oxadiazoline", 3),
    ("Amidoxime", 3),
    ("Simple coumarin", 3),
    ("Low-frequency / singleton classes", 10),
    ("Triazolo-thiadiazinyl", 10),
    ("Conjugated N-bearing coumarin", 47),
    ("Phosphonate coumarin", 5),
    ("Dimeric coumarin", 3),
    ("7-amino D–A", 2),
]

df = pd.DataFrame(data, columns=["Structural_class", "n"])

# ------------------------------------------------------------
# Save source data
# ------------------------------------------------------------
source_csv = OUT_SRC_DIR / "Figure_S1_structural_class_distribution_source_data.csv"
df.to_csv(source_csv, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# Plot settings (Q1-style clean publication figure)
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.4, 5.0))

bars = ax.barh(
    df["Structural_class"],
    df["n"],
    edgecolor="black",
    linewidth=0.8
)

# invert order so first category is on top
ax.invert_yaxis()

# annotate counts
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width)}",
        va="center",
        ha="left",
        fontsize=10.5
    )

# axis formatting
ax.set_xlabel("Number of compounds")
ax.set_ylabel("")
# ax.set_title("Structural-class distribution across the curated coumarin dataset", pad=10)

# limits and grid
xmax = max(df["n"]) + 6
ax.set_xlim(0, xmax)
ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.35)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

# clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ------------------------------------------------------------
# Save figure files
# ------------------------------------------------------------
png_path = OUT_FIG_DIR / "Figure_S1_structural_class_distribution.png"
pdf_path = OUT_FIG_DIR / "Figure_S1_structural_class_distribution.pdf"
svg_path = OUT_FIG_DIR / "Figure_S1_structural_class_distribution.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

plt.close(fig)

print("Figure S1 generated successfully.")
print(f"Source data : {source_csv}")
print(f"PNG         : {png_path}")
print(f"PDF         : {pdf_path}")
print(f"SVG         : {svg_path}")