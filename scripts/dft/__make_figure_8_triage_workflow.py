# -*- coding: utf-8 -*-
"""
Figure 8 FINAL — Prospective reliability-triage workflow

Purpose:
- Replace old FM-only taxonomy figure.
- Provide a practical prospective workflow for new coumarin analogues.
- Avoid text overflow by using controlled box heights, wrapped text, and compact labels.
- Output PNG, TIFF, PDF and CSV summary.

Author workflow:
Run from project root:
    python scripts\\make_figure_8_triage_workflow.py
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")

DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv",
    ROOT / "data" / "processed" / "benchmark_dataset.csv",
    ROOT / "Dataset_S1_benchmark_dataset.csv",
]

OUTPUT_DIR = ROOT / "figures" / "Figure8_prospective_triage_workflow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUTPUT_DIR / "Figure_8_prospective_triage_workflow.png"
FIG_TIFF = OUTPUT_DIR / "Figure_8_prospective_triage_workflow.tiff"
FIG_PDF = OUTPUT_DIR / "Figure_8_prospective_triage_workflow.pdf"
SUMMARY_CSV = OUTPUT_DIR / "Figure_8_triage_workflow_summary.csv"


def find_dataset() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset bulunamadı. Beklenen dosyalardan biri mevcut olmalı:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


INPUT_CSV = find_dataset()


# ============================================================
# 2. Dataset-derived FM summary
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_cols = ["Compound_ID", "FM", "delta_Consensus"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Dataset içinde eksik kolon(lar) var: {missing}")

df["delta_Consensus"] = pd.to_numeric(df["delta_Consensus"], errors="coerce")
df = df.dropna(subset=["FM", "delta_Consensus"]).copy()

fm_summary = (
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

fm_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

summary_dict = {
    row["FM"]: {
        "n": int(row["n"]),
        "mean_bias": float(row["mean_bias"]),
        "median_bias": float(row["median_bias"]),
        "mae": float(row["mae"]),
    }
    for _, row in fm_summary.iterrows()
}


# ============================================================
# 3. Plot settings
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# ============================================================
# 4. Helper functions
# ============================================================

def wrap_text(text: str, width: int) -> str:
    parts = []
    for paragraph in str(text).split("\n"):
        if paragraph.strip() == "":
            parts.append("")
        else:
            parts.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(parts)


def add_round_box(
    ax,
    x, y, w, h,
    facecolor="#FFFFFF",
    edgecolor="#BDBDBD",
    lw=1.0,
    radius=0.012,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def add_box(
    ax,
    x, y, w, h,
    title,
    body,
    facecolor="#FFFFFF",
    edgecolor="#BDBDBD",
    title_color="#111111",
    body_color="#111111",
    title_size=9.5,
    body_size=7.8,
    body_width=70,
    title_weight="bold",
):
    add_round_box(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor)

    ax.text(
        x + 0.014,
        y + h - 0.020,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight=title_weight,
        color=title_color,
    )

    if body:
        ax.text(
            x + 0.014,
            y + h - 0.052,
            wrap_text(body, body_width),
            ha="left",
            va="top",
            fontsize=body_size,
            color=body_color,
            linespacing=1.15,
        )


def arrow(ax, start, end, color="#666666", lw=1.1, mutation_scale=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
            connectionstyle="arc3,rad=0.0",
        )
    )


def get_fm(fm: str, key: str, default=None):
    return summary_dict.get(fm, {}).get(key, default)


def fmt_bias(value):
    if value is None:
        return "NA"
    return f"{value:+.2f}"


def add_fm_card(
    ax,
    x, y, w, h,
    fm,
    label,
    signal,
    action,
    color,
    face,
):
    add_round_box(ax, x, y, w, h, facecolor=face, edgecolor="#C7C7C7", lw=0.9, radius=0.010)
    ax.add_patch(Rectangle((x, y), 0.010, h, facecolor=color, edgecolor="none"))

    # Left FM label
    ax.text(
        x + 0.020,
        y + h - 0.019,
        fm,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=color,
    )

    ax.text(
        x + 0.020,
        y + h - 0.049,
        label,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="#111111",
    )

    # Middle signal
    ax.text(
        x + 0.195,
        y + h - 0.019,
        wrap_text(signal, 38),
        ha="left",
        va="top",
        fontsize=7.5,
        color="#111111",
        linespacing=1.12,
    )

    # Right action
    ax.text(
        x + w - 0.015,
        y + h - 0.022,
        wrap_text(action, 24),
        ha="right",
        va="top",
        fontsize=7.4,
        color="#111111",
        linespacing=1.10,
    )


# ============================================================
# 5. Canvas
# ============================================================

fig, ax = plt.subplots(figsize=(16.4, 9.4), dpi=600)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# ============================================================
# 6. Title
# ============================================================

ax.text(
    0.035,
    0.965,
    "FM0–FM4 failure-mode taxonomy and prospective reliability-triage workflow for coumarin logP prediction",
    ha="left",
    va="top",
    fontsize=15.5,
    fontweight="bold",
    color="#111111",
)

ax.text(
    0.035,
    0.925,
    "The framework prioritises when SwissADME-associated logP estimates should be trusted, treated as provisional, or replaced by experimental measurement.",
    ha="left",
    va="top",
    fontsize=9.7,
    color="#333333",
)

ax.text(0.035, 0.875, "A  Retrospective evidence base", fontsize=11.4, fontweight="bold", color="#111111")
ax.text(0.525, 0.875, "B  Prospective reliability-triage workflow", fontsize=11.4, fontweight="bold", color="#111111")

# Vertical divider
ax.plot([0.500, 0.500], [0.125, 0.850], color="#D0D0D0", linewidth=1.0)


# ============================================================
# 7. Panel A — Retrospective evidence base
# ============================================================

left_x = 0.035
left_w = 0.435

evidence_body = (
    "95-compound benchmark dataset\n"
    "Nitrogen-dependent bias reversal\n"
    "N-count × logP high-risk map\n"
    "Spread4 predictor-disagreement signal\n"
    "Documentation-tier sensitivity\n"
    "Outlier-robustness analysis\n"
    "Targeted DFT/MEP diagnostic panel"
)

add_box(
    ax,
    left_x, 0.690, left_w, 0.155,
    "Dataset-derived evidence",
    evidence_body,
    facecolor="#F7F7F7",
    edgecolor="#BDBDBD",
    title_size=9.8,
    body_size=7.8,
    body_width=58,
)

fm_cards = [
    {
        "fm": "FM0",
        "label": "Serviceable reference",
        "signal": f"N = 0, logP ≤ 3.0\nn = {get_fm('FM0', 'n', 6)}; mean bias = {fmt_bias(get_fm('FM0', 'mean_bias', -0.261))}",
        "action": "Use cautiously if predictors agree",
        "color": "#2E7D32",
        "face": "#EAF5EA",
    },
    {
        "fm": "FM1",
        "label": "Polar overestimation",
        "signal": f"N = 1–3, polar regime\nn = {get_fm('FM1', 'n', 13)}; mean bias = {fmt_bias(get_fm('FM1', 'mean_bias', -2.027))}",
        "action": "Prioritise experimental logP",
        "color": "#C62828",
        "face": "#FDEBEC",
    },
    {
        "fm": "FM2",
        "label": "Conjugated-N misassignment",
        "signal": f"N = 1–3, extended π\nn = {get_fm('FM2', 'n', 44)}; mean bias = {fmt_bias(get_fm('FM2', 'mean_bias', -0.872))}",
        "action": "Treat as provisional; check Spread4",
        "color": "#EF6C00",
        "face": "#FFF2E3",
    },
    {
        "fm": "FM3",
        "label": "Mixed high-N regime",
        "signal": f"N ≥ 4, mixed environments\nn = {get_fm('FM3', 'n', 27)}; reduced non-zero bias",
        "action": "Inspect individually",
        "color": "#9E7D00",
        "face": "#FFFBE8",
    },
    {
        "fm": "FM4",
        "label": "N-free π-extended boundary",
        "signal": f"N = 0, logP > 3.0\nn = {get_fm('FM4', 'n', 5)}; possible underestimation",
        "action": "Validate when logP matters",
        "color": "#1565C0",
        "face": "#E8F2FC",
    },
]

card_h = 0.077
card_gap = 0.012
card_y_start = 0.585

for i, card in enumerate(fm_cards):
    y = card_y_start - i * (card_h + card_gap)
    add_fm_card(
        ax,
        left_x, y, left_w, card_h,
        card["fm"], card["label"], card["signal"], card["action"],
        card["color"], card["face"],
    )


# ============================================================
# 8. Panel B — Prospective workflow
# ============================================================

right_x = 0.545
right_w = 0.420

workflow_boxes = [
    {
        "y": 0.765,
        "h": 0.080,
        "title": "New coumarin analogue",
        "body": "Start before experimental logP is available.",
        "face": "#F3F6FA",
        "edge": "#9AAEC2",
        "title_color": "#143B63",
        "body_width": 72,
        "body_size": 8.2,
    },
    {
        "y": 0.640,
        "h": 0.095,
        "title": "Pre-experimental inputs",
        "body": "Structure: N count, compact vs extended motif, donor–acceptor-conjugated embedding, π-extended dimeric character.\nSwissADME: consensus logP and Spread4.",
        "face": "#FFFFFF",
        "edge": "#BDBDBD",
        "title_color": "#111111",
        "body_width": 84,
        "body_size": 7.75,
    },
    {
        "y": 0.475,
        "h": 0.135,
        "title": "Reliability alerts",
        "body": "Compact oxadiazole-like N environment → lower concern.\nN = 1–3 + polar/conjugated motif → overestimation risk.\nN ≥ 4 mixed N environments → provisional.\nN-free π-extended dimer → possible underestimation boundary.",
        "face": "#FFFDF5",
        "edge": "#D2C28A",
        "title_color": "#6D5600",
        "body_width": 84,
        "body_size": 7.55,
    },
    {
        "y": 0.295,
        "h": 0.150,
        "title": "Decision output",
        "body": "Low concern: preliminary use acceptable if predictors agree.\nProvisional: inspect structure and predictor spread.\nHigh overestimation risk: do not deprioritise solely by predicted high logP.\nBoundary underestimation risk: do not assume low lipophilicity from prediction.",
        "face": "#F7FBFF",
        "edge": "#A9C5E5",
        "title_color": "#124F8C",
        "body_width": 86,
        "body_size": 7.35,
    },
    {
        "y": 0.150,
        "h": 0.105,
        "title": "Recommended action",
        "body": "Prioritise experimental logP measurement when predicted logP affects scaffold ranking, compound progression, or developability judgement.",
        "face": "#FDF1F1",
        "edge": "#D9A3A3",
        "title_color": "#8A1F1F",
        "body_width": 86,
        "body_size": 8.0,
    },
]

for b in workflow_boxes:
    add_box(
        ax,
        right_x, b["y"], right_w, b["h"],
        b["title"], b["body"],
        facecolor=b["face"],
        edgecolor=b["edge"],
        title_color=b["title_color"],
        title_size=9.8,
        body_size=b["body_size"],
        body_width=b["body_width"],
    )

# Vertical arrows inside Panel B
arrow(ax, (right_x + right_w / 2, 0.765), (right_x + right_w / 2, 0.735), color="#555555")
arrow(ax, (right_x + right_w / 2, 0.640), (right_x + right_w / 2, 0.610), color="#555555")
arrow(ax, (right_x + right_w / 2, 0.475), (right_x + right_w / 2, 0.445), color="#555555")
arrow(ax, (right_x + right_w / 2, 0.295), (right_x + right_w / 2, 0.255), color="#555555")

# Connector from Panel A to Panel B
arrow(ax, (0.470, 0.395), (0.540, 0.545), color="#777777", lw=1.0, mutation_scale=11)
ax.text(
    0.505,
    0.463,
    "translated into\nprospective triage",
    ha="center",
    va="center",
    fontsize=7.4,
    color="#555555",
)


# ============================================================
# 9. Footer
# ============================================================

footer = (
    "This workflow is a measurement-priority and reliability-triage framework. "
    "It is not a numerical logP correction equation, not a replacement for experimental logP, "
    "and not an independently trained classifier."
)

add_round_box(
    ax,
    0.035, 0.045, 0.930, 0.065,
    facecolor="#F7F7F7",
    edgecolor="#C8C8C8",
    lw=0.9,
    radius=0.012,
)

ax.text(
    0.055,
    0.077,
    wrap_text(footer, 180),
    ha="left",
    va="center",
    fontsize=8.1,
    color="#222222",
    linespacing=1.15,
)


# ============================================================
# 10. Export
# ============================================================

fig.savefig(FIG_PNG, dpi=600, bbox_inches="tight")
fig.savefig(FIG_TIFF, dpi=600, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
plt.close(fig)

print("\nFigure 8 prospective reliability-triage workflow generated successfully.")
print(f"Input dataset: {INPUT_CSV}")
print(f"PNG:           {FIG_PNG}")
print(f"TIFF:          {FIG_TIFF}")
print(f"PDF:           {FIG_PDF}")
print(f"Summary CSV:   {SUMMARY_CSV}")
print("\nDataset-derived FM summary:")
print(fm_summary.sort_values("FM").to_string(index=False))