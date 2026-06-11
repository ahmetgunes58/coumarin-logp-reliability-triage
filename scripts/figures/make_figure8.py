# -*- coding: utf-8 -*-
"""
Figure 8 FINAL
Failure-mode taxonomy for SwissADME-associated logP prediction in coumarin derivatives

Clean editorial version:
- compact taxonomy cards
- minimal text
- readable layout
- Table 10 carries the full detail
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import textwrap

# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(r"C:\Users\Ahmet Gunes\YandexDisk\Makaleler\4.1-\coumarin-logp\coumarin-logp")
INPUT_CSV = ROOT / "data" / "processed" / "benchmark_dataset.csv"

OUTPUT_DIR = ROOT / "figures" / "Figure8_failure_mode_taxonomy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUTPUT_DIR / "Figure8_failure_mode_taxonomy_FINAL.png"
FIG_TIFF = OUTPUT_DIR / "Figure8_failure_mode_taxonomy_FINAL.tiff"
FIG_PDF = OUTPUT_DIR / "Figure8_failure_mode_taxonomy_FINAL.pdf"
TABLE10_CSV = OUTPUT_DIR / "Table10_taxonomy_summary_FINAL.csv"


# ============================================================
# 2. Load and summarise data
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_cols = ["Compound_ID", "FM", "delta_Consensus"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["delta_Consensus"] = pd.to_numeric(df["delta_Consensus"], errors="coerce")
df = df.dropna(subset=["FM", "delta_Consensus"]).copy()

summary = (
    df.groupby("FM")
      .agg(
          n=("Compound_ID", "count"),
          mean_bias=("delta_Consensus", "mean"),
          median_bias=("delta_Consensus", "median"),
          mae=("delta_Consensus", lambda x: np.mean(np.abs(x))),
      )
      .round(3)
      .reset_index()
)

summary_dict = {
    row["FM"]: {
        "n": int(row["n"]),
        "mean_bias": float(row["mean_bias"]),
        "median_bias": float(row["median_bias"]),
        "mae": float(row["mae"]),
    }
    for _, row in summary.iterrows()
}


# ============================================================
# 3. Final compact taxonomy map
# ============================================================

taxonomy = {
    "FM0": {
        "label": "Low-risk / serviceable",
        "regime": "N = 0, logP ≤ 3.0",
        "meaning": "Fragment locality retained",
        "action": "Use if predictors agree",
        "fill": "#EAF5EA",
        "bar": "#2E7D32",
        "accent": "#1B5E20",
    },
    "FM1": {
        "label": "Polar overestimation",
        "regime": "N = 1–3, logP < 1.5",
        "meaning": "Strongest overestimation;\ndonor–π–acceptor embedding",
        "action": "Measure experimentally",
        "fill": "#FDEBEC",
        "bar": "#C62828",
        "accent": "#B71C1C",
    },
    "FM2": {
        "label": "Conjugated-N misassignment",
        "regime": "N = 1–3, extended π,\nlogP > 1.5",
        "meaning": "Cross-fragment coupling\ncauses directional bias",
        "action": "Treat as provisional;\ncheck Spread4",
        "fill": "#FFF2E3",
        "bar": "#EF6C00",
        "accent": "#E65100",
    },
    "FM3": {
        "label": "High-N cancellation",
        "regime": "N ≥ 4, mixed N\nenvironments",
        "meaning": "Cancellation can mask\nstructure-specific failure",
        "action": "Inspect individually",
        "fill": "#FFFBE8",
        "bar": "#9E7D00",
        "accent": "#7A6100",
    },
    "FM4": {
        "label": "Conjugation overflow",
        "regime": "N = 0, logP > 3.0",
        "meaning": "Opposite-bias regime\nwith underestimation risk",
        "action": "Validate when logP matters",
        "fill": "#E8F2FC",
        "bar": "#1565C0",
        "accent": "#0D47A1",
    },
}

for fm in taxonomy:
    taxonomy[fm].update(summary_dict[fm])


# ============================================================
# 4. Export helper CSV for Table 10 cross-check
# ============================================================

table10_export = pd.DataFrame([
    {
        "Mode": fm,
        "Label": taxonomy[fm]["label"],
        "Representative regime": taxonomy[fm]["regime"].replace("\n", " "),
        "n": taxonomy[fm]["n"],
        "Mean bias": taxonomy[fm]["mean_bias"],
        "Median bias": taxonomy[fm]["median_bias"],
        "MAE": taxonomy[fm]["mae"],
        "Mechanistic meaning": taxonomy[fm]["meaning"].replace("\n", " "),
        "Recommended action": taxonomy[fm]["action"].replace("\n", " "),
    }
    for fm in ["FM0", "FM1", "FM2", "FM3", "FM4"]
])

table10_export.to_csv(TABLE10_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 5. Plot settings
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(11.6, 8.0), dpi=600)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# ============================================================
# 6. Helper functions
# ============================================================

def add_round_rect(ax, x, y, w, h, facecolor, edgecolor="#C8C8C8", lw=0.9, radius=0.013):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, text, size=9, weight="normal", color="black",
             ha="left", va="center", style=None):
    ax.text(
        x, y, text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        style=style,
    )


def bias_word(mean_bias):
    return "overestimation" if mean_bias < 0 else "underestimation"


def draw_card(ax, x, y, w, h, fm, meta):
    # Card
    add_round_rect(ax, x, y, w, h, facecolor=meta["fill"], edgecolor="#C7C7C7", lw=0.9, radius=0.013)
    ax.add_patch(Rectangle((x, y), 0.013, h, facecolor=meta["bar"], edgecolor="none"))

    # Column anchors
    x_mode = x + 0.022
    x_reg  = x + 0.27
    x_mean = x + 0.54
    x_act  = x + 0.79

    # Left block: mode
    add_text(ax, x_mode, y + h*0.66, fm, size=15, weight="bold", color=meta["accent"])
    add_text(ax, x_mode, y + h*0.34, meta["label"], size=9.6, weight="bold", color="#111111")

    # Regime + stats
    add_text(ax, x_reg, y + h*0.62, meta["regime"], size=9.5, color="#111111")
    add_text(ax, x_reg, y + h*0.34, f"n = {meta['n']}   |   mean bias = {meta['mean_bias']:+.3f}", size=9.3, color="#111111")
    add_text(ax, x_reg, y + h*0.14, bias_word(meta["mean_bias"]), size=9.1, color=meta["accent"], style="italic")

    # Mechanistic meaning
    add_text(ax, x_mean, y + h*0.50, meta["meaning"], size=9.3, color="#111111")

    # Action
    add_text(ax, x_act, y + h*0.50, meta["action"], size=9.1, color="#111111")


# ============================================================
# 7. Title / subtitle / section hints
# ============================================================

add_text(
    ax,
    0.03,
    0.965,
    "Failure-mode taxonomy for SwissADME-associated logP prediction in coumarin derivatives",
    size=15,
    weight="bold",
    va="top",
)

subtitle = (
    "Compact decision-oriented summary of FM0–FM4. "
    "Negative mean bias indicates overestimation of lipophilicity "
    "(experimental logP − predicted logP < 0)."
)
add_text(ax, 0.03, 0.925, "\n".join(textwrap.wrap(subtitle, width=120)), size=9.4, color="#222222", va="top")

add_text(ax, 0.25, 0.875, "Regime and dataset signal", size=9.3, weight="bold", color="#444444")
add_text(ax, 0.55, 0.875, "Mechanistic meaning", size=9.3, weight="bold", color="#444444")
add_text(ax, 0.80, 0.875, "Recommended action", size=9.3, weight="bold", color="#444444")
ax.plot([0.03, 0.97], [0.858, 0.858], color="#BDBDBD", linewidth=0.9)


# ============================================================
# 8. Cards
# ============================================================

x = 0.03
w = 0.94
h = 0.110
gap = 0.020
top_y = 0.72

order = ["FM0", "FM1", "FM2", "FM3", "FM4"]

for i, fm in enumerate(order):
    y = top_y - i * (h + gap)
    draw_card(ax, x, y, w, h, fm, taxonomy[fm])


# ============================================================
# 9. Footer
# ============================================================

footer = (
    "FM0 is the serviceable reference regime. FM1 and FM2 are dominated by overestimation, "
    "FM3 reflects cancellation among heterogeneous high-nitrogen environments, and FM4 is the "
    "opposite-bias underestimation regime. The taxonomy is a risk-recognition framework, not a replacement "
    "for experimental logP measurement."
)

add_round_rect(
    ax,
    0.03,
    0.032,
    0.94,
    0.062,
    facecolor="#F7F7F7",
    edgecolor="#C8C8C8",
    lw=0.8,
    radius=0.012
)
add_text(ax, 0.045, 0.063, "\n".join(textwrap.wrap(footer, width=160)), size=8.4, color="#222222")


# ============================================================
# 10. Export
# ============================================================

fig.savefig(FIG_PNG, dpi=600, bbox_inches="tight")
fig.savefig(FIG_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
plt.close(fig)

print("\nFigure 8 FINAL generated successfully.")
print(f"Input dataset: {INPUT_CSV}")
print(f"PNG:           {FIG_PNG}")
print(f"TIFF:          {FIG_TIFF}")
print(f"PDF:           {FIG_PDF}")
print(f"Table10 CSV:   {TABLE10_CSV}")
print("\nDataset-derived FM summary:")
print(summary.sort_values('FM').to_string(index=False))