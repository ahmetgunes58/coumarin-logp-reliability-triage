from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# ============================================================
# Figure 5
# Nitrogen-count × experimental logP high-risk map
# Final main-text version
# ============================================================

# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_5_ncount_logP_high_risk_map.png"
OUT_PDF = OUT_DIR / "Figure_5_ncount_logP_high_risk_map.pdf"
OUT_TIF = OUT_DIR / "Figure_5_ncount_logP_high_risk_map.tif"

# ---------- Data ----------
# Rows: N = 0, N = 1–3, N ≥ 4
# Cols: logP < 1.5, 1.5 ≤ logP < 3.0, logP ≥ 3.0

bias_matrix = np.array([
    [np.nan, -0.23,  0.75],
    [-2.03, -0.75, -1.05],
    [-1.78, -0.24,  0.37]
], dtype=float)

n_matrix = np.array([
    [1, 5, 5],
    [13, 26, 18],
    [11, 12, 4]
], dtype=int)

x_labels = ["logP < 1.5", "1.5 ≤ logP ≤ 3.0", "logP > 3.0"]
y_labels = ["N = 0", "N = 1–3", "N ≥ 4"]

# ---------- Figure style ----------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14
})

fig, ax = plt.subplots(figsize=(8.4, 6.2))

# Colormap: negative = blue, positive = red
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad("#dfe7ee")  # for the "not interpreted" cell

norm = TwoSlopeNorm(vmin=-2.2, vcenter=0.0, vmax=1.0)
im = ax.imshow(bias_matrix, cmap=cmap, norm=norm, aspect="equal")

# ---------- Axes ----------
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

ax.set_xlabel("Experimental logP range")
ax.set_ylabel("Nitrogen-count group")

# Remove outer spines for a cleaner Q1-style appearance
for spine in ax.spines.values():
    spine.set_visible(False)

# White gridlines between cells
ax.set_xticks(np.arange(-0.5, bias_matrix.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, bias_matrix.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
ax.tick_params(which="minor", bottom=False, left=False)

# ---------- Highlight principal high-risk cell ----------
# Cell: row 1 (N = 1–3), col 0 (logP < 1.5)
highlight = Rectangle(
    (-0.5, 0.5), 1, 1,
    fill=False, edgecolor="black", linewidth=2.8
)
ax.add_patch(highlight)

# ---------- Helper for text color ----------
def text_color_from_value(value):
    """Choose white text for dark cells and black text for light cells."""
    if np.isnan(value):
        return "black"
    rgba = cmap(norm(value))
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.52 else "black"

# ---------- Cell annotations ----------
for i in range(bias_matrix.shape[0]):
    for j in range(bias_matrix.shape[1]):
        value = bias_matrix[i, j]
        n_val = n_matrix[i, j]

        if i == 0 and j == 0:
            # not interpreted cell
            ax.text(
                j, i,
                f"n = {n_val}\nnot interpreted",
                ha="center", va="center",
                color="black",
                fontsize=12
            )
        elif i == 1 and j == 0:
            # principal high-risk zone cell
            ax.text(
                j, i,
                f"principal\nhigh-risk\nzone\n{value:+.2f}\n(n = {n_val})",
                ha="center", va="center",
                color="white",
                fontsize=13,
                fontweight="bold",
                linespacing=1.15
            )
        else:
            ax.text(
                j, i,
                f"{value:+.2f}\n(n = {n_val})",
                ha="center", va="center",
                color=text_color_from_value(value),
                fontsize=13
            )

# ---------- Colorbar ----------
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Mean consensus bias, ΔlogP", rotation=90, labelpad=16)
cbar.ax.tick_params(labelsize=11)

# ---------- Layout ----------
plt.subplots_adjust(left=0.14, right=0.88, bottom=0.16, top=0.98)

# ---------- Save ----------
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
fig.savefig(OUT_TIF, dpi=600, bbox_inches="tight")
plt.close(fig)

print("Figure 5 generated successfully.")
print(f"PNG : {OUT_PNG}")
print(f"PDF : {OUT_PDF}")
print(f"TIF : {OUT_TIF}")