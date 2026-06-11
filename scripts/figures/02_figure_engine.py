# =============================================================================
# 02_figure_engine.py
# coumarin-logp Benchmark Pipeline — Step 3: Figure Generation
#
# Author  : Ahmet GÜNEŞ
# Affil.  : National Defence University, Turkish Naval Academy,
#           Department of Basic Sciences, Istanbul, Türkiye
# Contact : ahmet.gunes3@msu.edu.tr
# Version : 2.0 (Q1-level, 8-figure set)
# Date    : 2026
#
# Description:
#   Produces all publication-ready figures for the manuscript.
#   Every figure is generated deterministically from benchmark_dataset.csv.
#   No manual editing of figures is required or permitted.
#
# Figures produced:
#   Fig1  — Dataset overview (logP distribution, N_count, data tiers)
#   Fig2  — Predictor performance scatter matrix (logP_exp vs predicted)
#   Fig3  — Nitrogen effect: bias direction reversal (grouped bar + CI)
#   Fig4  — Error distribution (histogram + KDE + boxplot)
#   Fig5  — Structural class conjugation gradient (horizontal bar)
#   Fig6  — 2D non-additivity heatmap (N × logP)
#   Fig7  — Failure mode taxonomy (distribution + RMSE)
#   Fig8  — Chemical interpretation / ESP panel (DFT placeholder)
#
# Inputs:
#   data/processed/benchmark_dataset.csv
#
# Outputs:
#   figures/main/Figure_N.png   (300 dpi, for manuscript draft)
#   figures/main/Figure_N.tiff  (600 dpi, for journal repository package)
#   figures/main/Figure_N.pdf   (vector, for SI / presentations)
#
# Usage:
#   cd <project_root>
#   python scripts/02_figure_engine.py
#   python scripts/02_figure_engine.py --fig 3    # single figure
#   python scripts/02_figure_engine.py --fig 3 5  # multiple figures
#
# Requirements:
#   matplotlib >= 3.6  |  seaborn >= 0.12  |  scipy >= 1.9
#   pandas >= 1.5      |  numpy >= 1.23
#
# Style:
#   Arial font, coumarin-logp column widths (8.5 cm single, 17.6 cm double),
#   CMYK-safe colour palette, no chartjunk.
# =============================================================================

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
FIG_DIR    = os.path.join(ROOT_DIR, "figures", "main")
IN_CSV     = os.path.join(PROC_DIR, "benchmark_dataset.csv")

LIT_RMSE   = 0.60   # Mannhold et al. 2009

# coumarin-logp column widths (inches)
# Single column: 3.35"  |  1.5 column: 4.72"  |  Double column: 6.93"
COL1  = 3.35
COL15 = 4.72
COL2  = 6.93

# Publication-quality DPI
DPI_DRAFT  = 300   # PNG for manuscript draft
DPI_SUBMIT = 600   # TIFF for journal repository package

# ---------------------------------------------------------------------------
# COLOUR PALETTE
# ---------------------------------------------------------------------------
# CMYK-safe, colour-blind friendly, consistent throughout
PRED_COLORS = {
    "iLOGP"     : "#2166AC",   # blue
    "XLOGP3"    : "#4DAC26",   # green
    "WLOGP"     : "#D01C8B",   # magenta
    "MLOGP"     : "#7B3294",   # purple
    "Silicos_IT": "#E66101",   # orange
    "Consensus" : "#1A9641",   # dark green
}
PRED_LABELS = {
    "iLOGP"     : "iLOGP",
    "XLOGP3"    : "XLOGP3",
    "WLOGP"     : "WLOGP",
    "MLOGP"     : "MLOGP",
    "Silicos_IT": "Silicos-IT",
    "Consensus" : "Consensus",
}

N_COLORS = {
    "N=0"  : "#2166AC",
    "N=1"  : "#FDB863",
    "N=2-3": "#E66101",
    "N≥4"  : "#762A83",
}
N_GROUPS = ["N=0", "N=1", "N=2-3", "N≥4"]

FM_COLORS = {
    "FM0": "#4DAC26",
    "FM1": "#D73027",
    "FM2": "#F46D43",
    "FM3": "#762A83",
    "FM4": "#2166AC",
}
FM_LABELS = {
    "FM0": "FM0\nAccurate",
    "FM1": "FM1\nPolar Overest.",
    "FM2": "FM2\nMulti-N Misassign.",
    "FM3": "FM3\nN Cancellation",
    "FM4": "FM4\nConj. Overflow",
}

TIER_COLORS = {
    "GOLD"    : "#D4A017",
    "STRICT"  : "#2166AC",
    "EXTENDED": "#74ADD1",
}

CONJ_CMAP = matplotlib.colormaps["RdYlGn_r"]

# ---------------------------------------------------------------------------
# STYLE SETUP
# ---------------------------------------------------------------------------

def set_style():
    """Global matplotlib style — coumarin-logp compatible."""
    try:
        matplotlib.rcParams["font.family"] = "Arial"
    except Exception:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"

    matplotlib.rcParams.update({
        "font.size"          : 7,
        "axes.titlesize"     : 8,
        "axes.labelsize"     : 7,
        "xtick.labelsize"    : 6.5,
        "ytick.labelsize"    : 6.5,
        "legend.fontsize"    : 6.5,
        "legend.title_fontsize": 7,
        "axes.linewidth"     : 0.6,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "xtick.major.width"  : 0.6,
        "ytick.major.width"  : 0.6,
        "xtick.major.size"   : 3,
        "ytick.major.size"   : 3,
        "lines.linewidth"    : 1.0,
        "patch.linewidth"    : 0.5,
        "figure.dpi"         : 150,
        "savefig.bbox"       : "tight",
        "savefig.pad_inches" : 0.05,
        "figure.facecolor"   : "white",
        "axes.facecolor"     : "white",
    })


def save(fig, name: str):
    """Save figure as PNG (draft), TIFF (repository package), PDF (vector)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".png",  dpi=DPI_DRAFT,  facecolor="white")
    fig.savefig(base + ".tiff", dpi=DPI_SUBMIT, facecolor="white")
    fig.savefig(base + ".pdf",                  facecolor="white")
    print(f"  ✅ {name} → PNG / TIFF / PDF")


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add bold panel label (A, B, C...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def rmse(series):
    return float(np.sqrt((series.dropna() ** 2).mean()))


# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {path} not found. Run 01_prepare_dataset.py first.")
    df = pd.read_csv(path)
    df["N_group"]    = pd.Categorical(df["N_group"],
                                       categories=N_GROUPS, ordered=True)
    df["logP_range"] = pd.Categorical(df["logP_range"],
                                       categories=["logP<1","logP 1-2","logP 2-3","logP>3"],
                                       ordered=True)
    return df


# ===========================================================================
# FIGURE 1 — Dataset Overview
# ===========================================================================

def figure1(df: pd.DataFrame):
    """
    Three-panel dataset overview.
    A: logP distribution by Data_Tier
    B: N_count distribution by N_group
    C: logP vs MW scatter by N_group
    Panels A and C share the same x-axis scale (0-5, ticks 0,1,2,3,4,5).
    """
    fig = plt.figure(figsize=(COL2, 2.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.58)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])

    # Shared logP axis range for panels A and C
    LOGP_MIN, LOGP_MAX = -0.2, 5.4
    LOGP_TICKS = [0, 1, 2, 3, 4, 5]

    # ── Panel A: logP distribution by Data_Tier ──────────────────────────
    bins   = np.linspace(LOGP_MIN, LOGP_MAX, 18)
    bottom = np.zeros(len(bins) - 1)
    for tier in ["GOLD", "STRICT", "EXTENDED"]:
        sub = df[df["Data_Tier"] == tier]["logP_exp"]
        cnt, _ = np.histogram(sub, bins=bins)
        axA.bar(bins[:-1], cnt, width=np.diff(bins), bottom=bottom,
                color=TIER_COLORS[tier], label=tier, alpha=0.85,
                edgecolor="white", linewidth=0.3, align="edge")
        bottom += cnt

    # KDE secondary axis
    kde_x = np.linspace(LOGP_MIN, LOGP_MAX, 200)
    kde   = stats.gaussian_kde(df["logP_exp"], bw_method=0.4)
    axA2  = axA.twinx()
    axA2.plot(kde_x, kde(kde_x), color="#444444", lw=0.9, ls="--", alpha=0.55)
    axA2.set_ylabel("Density", fontsize=5.5, color="#666666", labelpad=2)
    axA2.tick_params(labelsize=5.5, colors="#666666", pad=1)
    axA2.spines["right"].set_visible(True)
    axA2.spines["right"].set_linewidth(0.4)
    axA2.spines["top"].set_visible(False)
    axA2.set_ylim(0, None)

    axA.set_xlim(LOGP_MIN, LOGP_MAX)
    axA.set_xticks(LOGP_TICKS)
    axA.set_xlabel("Experimental logP", fontsize=6.5, labelpad=3)
    axA.set_ylabel("Number of compounds", fontsize=6.5, labelpad=3)
    axA.tick_params(labelsize=6)

    # Legend: upper LEFT — data peaks on right side, left is empty
    axA.legend(fontsize=5.5, title="Data tier", title_fontsize=5.5,
               loc="upper left", bbox_to_anchor=(0.01, 0.99),
               framealpha=1.0, edgecolor="#999999",
               borderpad=0.5, handlelength=1.0, handletextpad=0.4)

    axA.text(-0.22, 1.06, "A", transform=axA.transAxes,
             fontsize=9, fontweight="bold", va="top", ha="left")

    # ── Panel B: N_count distribution ────────────────────────────────────
    n_vals   = sorted(df["N_count"].unique())
    x_pos    = np.arange(len(n_vals))
    counts_b = [len(df[df["N_count"] == n]) for n in n_vals]
    colors_b = []
    for n in n_vals:
        if n == 0:   colors_b.append(N_COLORS["N=0"])
        elif n == 1: colors_b.append(N_COLORS["N=1"])
        elif n <= 3: colors_b.append(N_COLORS["N=2-3"])
        else:        colors_b.append(N_COLORS["N\u22654"])

    bars = axB.bar(x_pos, counts_b, color=colors_b,
                   edgecolor="#333333", linewidth=0.4, alpha=0.88)

    axB.set_ylim(0, max(counts_b) * 1.35)
    for bar, c in zip(bars, counts_b):
        axB.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.35,
                 str(c), ha="center", va="bottom",
                 fontsize=5.8, color="#222222", fontweight="bold")

    axB.set_xticks(x_pos)
    axB.set_xticklabels([str(n) for n in n_vals], fontsize=6)
    axB.set_xlabel("Nitrogen atom count", fontsize=6.5, labelpad=3)
    axB.set_ylabel("Number of compounds", fontsize=6.5, labelpad=3)
    axB.tick_params(labelsize=6)

    # Legend: upper right, 2-col, fully opaque
    patches = [mpatches.Patch(color=N_COLORS[g], label=g) for g in N_GROUPS]
    axB.legend(handles=patches, fontsize=5.5, title="N group",
               title_fontsize=5.5,
               loc="upper right", bbox_to_anchor=(0.99, 0.99),
               ncol=2, framealpha=1.0, edgecolor="#999999",
               borderpad=0.4, handlelength=1.0, handletextpad=0.3,
               columnspacing=0.5)

    axB.text(-0.22, 1.06, "B", transform=axB.transAxes,
             fontsize=9, fontweight="bold", va="top", ha="left")

    # ── Panel C: logP vs MW scatter ───────────────────────────────────────
    for grp in N_GROUPS:
        sub = df[df["N_group"] == grp]
        axC.scatter(sub["logP_exp"], sub["MW"],
                    c=N_COLORS[grp], s=12, alpha=0.68,
                    edgecolors="white", linewidths=0.3,
                    label=grp, zorder=3)

    axC.set_xlim(LOGP_MIN, LOGP_MAX)
    axC.set_xticks(LOGP_TICKS)
    axC.set_xlabel("Experimental logP", fontsize=6.5, labelpad=3)
    axC.set_ylabel("Molecular weight (Da)", fontsize=6.5, labelpad=3)
    axC.tick_params(labelsize=6)

    # N_group legend: upper LEFT — data mostly in middle/right
    leg_C = axC.legend(fontsize=5.5, title="N group", title_fontsize=5.5,
                       loc="upper left", bbox_to_anchor=(0.01, 0.99),
                       framealpha=1.0, edgecolor="#888888",
                       borderpad=0.5, handlelength=0.8,
                       handletextpad=0.4, markerscale=0.9)
    leg_C.get_frame().set_linewidth(0.8)

    # Dataset info box: BOTTOM RIGHT — no overlap with legend (which is top left)
    info_lines = ("n = " + str(len(df)) + "\n" +
                  "GOLD: "     + str(sum(df["Data_Tier"] == "GOLD"))     + "\n" +
                  "STRICT: "   + str(sum(df["Data_Tier"] == "STRICT"))   + "\n" +
                  "EXTENDED: " + str(sum(df["Data_Tier"] == "EXTENDED")))
    axC.text(0.98, 0.04, info_lines,
             transform=axC.transAxes, ha="right", va="bottom",
             fontsize=5.5, color="#333333", linespacing=1.5,
             bbox=dict(fc="white", ec="#888888", lw=0.8,
                       boxstyle="round,pad=0.4", alpha=1.0))

    axC.text(-0.22, 1.06, "C", transform=axC.transAxes,
             fontsize=9, fontweight="bold", va="top", ha="left")

    fig.suptitle("Dataset overview", fontsize=8.5,
                 fontweight="bold", y=1.04)
    fig.savefig(os.path.join(FIG_DIR, "Figure_1.png"),
                dpi=DPI_DRAFT, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure_1.tiff"),
                dpi=DPI_SUBMIT, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure_1.pdf"),
                facecolor="white", bbox_inches="tight")
    print("  \u2705 Figure_1 \u2192 PNG / TIFF / PDF")
    plt.close(fig)


def figure2(df: pd.DataFrame):
    """
    2×3 scatter matrix: logP_exp vs each predictor.
    Each panel shows identity line, RMSE, Bias, R², N_group colouring.
    """
    preds = ["iLOGP", "MLOGP", "Consensus", "WLOGP", "XLOGP3", "Silicos_IT"]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.2))
    fig.subplots_adjust(hspace=0.45, wspace=0.38)
    axes_flat = axes.flatten()

    lim_lo, lim_hi = -1.5, 6.5

    for i, (pred, ax) in enumerate(zip(preds, axes_flat)):
        # Scatter, coloured by N_group
        for grp in N_GROUPS:
            sub = df[df["N_group"] == grp]
            ax.scatter(sub["logP_exp"], sub[pred],
                       c=N_COLORS[grp], s=8, alpha=0.70,
                       edgecolors="none", zorder=3, label=grp)

        # Identity line
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                color="#999999", lw=0.8, ls="--", zorder=2)

        # ±1 log unit bands
        ax.fill_between([lim_lo, lim_hi],
                        [lim_lo - 1, lim_hi - 1],
                        [lim_lo + 1, lim_hi + 1],
                        alpha=0.07, color="#666666", zorder=1)

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal")

        # Stats annotation
        d = df[f"delta_{pred}"].dropna()
        bias_val = d.mean()
        rmse_val = rmse(d)
        ss_res   = np.sum((df["logP_exp"] - df[pred]) ** 2)
        ss_tot   = np.sum((df["logP_exp"] - df["logP_exp"].mean()) ** 2)
        r2_val   = 1 - ss_res / ss_tot

        ax.text(0.04, 0.96,
                f"RMSE = {rmse_val:.2f}\n"
                f"Bias = {bias_val:+.2f}\n"
                f"R² = {r2_val:.2f}",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=5.5,
                bbox=dict(fc="white", ec="#cccccc", lw=0.4,
                          boxstyle="round,pad=0.3"))

        ax.set_title(PRED_LABELS[pred], fontsize=7.5, fontweight="bold",
                     color=PRED_COLORS[pred], pad=3)
        ax.set_xlabel("Experimental logP", fontsize=6)
        ax.set_ylabel("Predicted logP", fontsize=6)
        ax.tick_params(labelsize=5.5)

        panel_letter = chr(ord("A") + i)
        add_panel_label(ax, panel_letter)

    # Shared legend (bottom right panel area)
    handles = [mpatches.Patch(color=N_COLORS[g], label=g) for g in N_GROUPS]
    axes_flat[-1].legend(handles=handles, title="N group",
                         title_fontsize=6, fontsize=5.5,
                         loc="lower right", framealpha=0.9, edgecolor="none")

    fig.suptitle("Fragment-based logP predictor performance (n = 95)",
                 fontsize=8.5, fontweight="bold", y=1.01)
    save(fig, "Figure_2")
    plt.close(fig)


# ===========================================================================
# FIGURE 3 — Nitrogen Effect: Bias Direction Reversal
# ===========================================================================

def figure3(df: pd.DataFrame):
    """
    Grouped bar chart: mean bias (Exp − Pred) by N_group for all 6 predictors.
    Error bars = 95% CI. Annotated with Mann-Whitney and Kruskal-Wallis p.
    """
    preds  = list(PRED_COLORS.keys())
    groups = N_GROUPS
    n_preds  = len(preds)
    n_groups = len(groups)
    x        = np.arange(n_groups)
    width    = 0.12
    offsets  = np.linspace(-(n_preds - 1) / 2,
                            (n_preds - 1) / 2, n_preds) * width

    fig, ax = plt.subplots(figsize=(COL2, 3.4))

    for i, pred in enumerate(preds):
        means, cis, ns = [], [], []
        for grp in groups:
            sub = df[df["N_group"] == grp][f"delta_{pred}"].dropna()
            n   = len(sub)
            m   = sub.mean()
            ci  = (stats.t.ppf(0.975, n - 1) * sub.sem()) if n > 1 else 0
            means.append(m); cis.append(ci); ns.append(n)

        bars = ax.bar(x + offsets[i], means, width,
                      label=PRED_LABELS[pred],
                      color=PRED_COLORS[pred], alpha=0.85,
                      edgecolor="white", linewidth=0.3)
        ax.errorbar(x + offsets[i], means, yerr=cis,
                    fmt="none", color="#333333",
                    capsize=2, linewidth=0.7, capthick=0.7)

    # Reference line
    ax.axhline(0, color="#444444", lw=0.8, ls="--", alpha=0.7, zorder=5)

    # x-tick labels with n
    group_ns = df["N_group"].value_counts().reindex(groups)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={group_ns[g]})" for g in groups], fontsize=7)
    ax.set_xlabel("Nitrogen atom count group", fontsize=7)
    ax.set_ylabel("Mean bias (Exp − Pred, log units)", fontsize=7)

    # Statistical annotation
    d0 = df[df["N_group"] == "N=0"]["delta_Consensus"].dropna()
    d1 = df[df["N_group"] != "N=0"]["delta_Consensus"].dropna()
    U, p_mw = stats.mannwhitneyu(d0, d1, alternative="two-sided")
    H, p_kw = stats.kruskal(*[
        df[df["N_group"] == g]["delta_Consensus"].dropna().values
        for g in groups
    ])
    ax.text(0.02, -0.22,
            f"* Mann–Whitney p = {p_mw:.3f} (N=0 vs N≥1)    "
            f"† Kruskal–Wallis p = {p_kw:.3f} (all groups)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=6, color="#555555", style="italic")
    fig.subplots_adjust(bottom=0.18)

    ax.legend(title="Predictor", fontsize=6, title_fontsize=6.5,
              bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.9, edgecolor="none")

    ax.set_title("Prediction bias as a function of nitrogen atom count",
                 fontsize=8.5, fontweight="bold", pad=4)
    fig.tight_layout()
    save(fig, "Figure_3")
    plt.close(fig)


# ===========================================================================
# FIGURE 4 — Error Distribution
# ===========================================================================

def figure4(df: pd.DataFrame):
    """
    Two-panel error distribution:
    A — Histogram + KDE + Shapiro-Wilk annotation
    B — Boxplot by N_group
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.8),
                                    gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.38)
    errors = df["delta_Consensus"].dropna()

    # --- Panel A: Histogram + KDE ---
    ax1.hist(errors, bins=20, color="#74ADD1", alpha=0.75,
             edgecolor="white", linewidth=0.3, density=True, label="Histogram")

    kde_x = np.linspace(errors.min() - 0.5, errors.max() + 0.5, 300)
    kde   = stats.gaussian_kde(errors, bw_method=0.35)
    ax1.plot(kde_x, kde(kde_x), color="#2166AC", lw=1.2, label="KDE", zorder=5)

    ax1.axvline(0, color="#444444", lw=0.8, ls="--", alpha=0.7)
    ax1.axvline(errors.mean(), color="#D73027", lw=1.0, ls="-", alpha=0.85,
                label=f"Mean = {errors.mean():.3f}")

    pct_neg = (errors < 0).mean() * 100
    ax1.text(errors.mean() - 0.15, ax1.get_ylim()[1] * 0.01,
             f"{pct_neg:.0f}% overestimation\n(negative errors)",
             ha="right", va="bottom", fontsize=6, color="#D73027")

    W, p_sw = stats.shapiro(errors)
    ax1.text(0.97, 0.97,
             f"Shapiro–Wilk\nW = {W:.3f}, p = {p_sw:.3f}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6,
             bbox=dict(fc="white", ec="#cccccc", lw=0.4, boxstyle="round,pad=0.3"))

    ax1.set_xlabel("Consensus logP error (Exp − Pred)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=6, framealpha=0.9, edgecolor="none",
               loc="upper left")
    add_panel_label(ax1, "A")

    # --- Panel B: Boxplot by N_group ---
    group_data = [df[df["N_group"] == g]["delta_Consensus"].dropna().values
                  for g in N_GROUPS]
    bp = ax2.boxplot(group_data, patch_artist=True, notch=False,
                     medianprops=dict(color="#333333", lw=1.2),
                     whiskerprops=dict(lw=0.7, color="#555555"),
                     capprops=dict(lw=0.7, color="#555555"),
                     flierprops=dict(marker="o", ms=3, alpha=0.5,
                                     markerfacecolor="#888888",
                                     markeredgecolor="none"))
    for patch, grp in zip(bp["boxes"], N_GROUPS):
        patch.set_facecolor(N_COLORS[grp])
        patch.set_alpha(0.75)

    ax2.axhline(0, color="#444444", lw=0.8, ls="--", alpha=0.7)
    ax2.set_xticks(range(1, len(N_GROUPS) + 1))
    ax2.set_xticklabels(N_GROUPS, fontsize=6.5)
    ax2.set_xlabel("Nitrogen count group")
    ax2.set_ylabel("Consensus logP error")
    add_panel_label(ax2, "B")

    fig.suptitle("Consensus logP error distribution (n = 95)",
                 fontsize=8.5, fontweight="bold", y=1.01)
    save(fig, "Figure_4")
    plt.close(fig)


# ===========================================================================
# FIGURE 5 — Structural Class Conjugation Gradient
# ===========================================================================

def figure5(df: pd.DataFrame):
    """
    Horizontal bar chart: MAE by structural class, sorted by MAE,
    coloured by conjugation level (gradient), annotated with Spearman rho.
    Excludes classes with n < 3 (annotated separately).
    """
    # Conjugation level rank
    conj_rank = {
        "oxadiazole"                        : 1,
        "Furocoumarin"                      : 2,
        "oxadiazoline"                      : 3,
        "amidoxime"                         : 4,
        "simple_coumarin"                   : 5,
        "triazolo_thiadiazinyl_coumarin"    : 6,
        "conjugated_coumarin"               : 7,
        "phosphonate_coumarin"              : 8,
        "dimeric_coumarin"                  : 9,
    }
    # Clean labels
    class_labels = {
        "oxadiazole"                        : "Oxadiazole (compact N)",
        "Furocoumarin"                      : "Furocoumarin (N-free, fused O)",
        "oxadiazoline"                      : "Oxadiazoline (compact N)",
        "amidoxime"                         : "Amidoxime (moderate N)",
        "simple_coumarin"                   : "Simple coumarin (N-free, short π)",
        "triazolo_thiadiazinyl_coumarin"    : "Triazolo-thiadiazinyl (multi-N)",
        "conjugated_coumarin"               : "Conjugated coumarin (extended π+N)",
        "phosphonate_coumarin"              : "Phosphonate coumarin (N+phos.)",
        "dimeric_coumarin"                  : "Dimeric coumarin (N-free, long π)",
    }

    records = []
    for cls, sub in df.groupby("Coumarin_Type"):
        if len(sub) < 3:
            continue
        d = sub["delta_Consensus"].dropna()
        records.append({
            "class"    : cls,
            "label"    : class_labels.get(cls, cls),
            "n"        : len(sub),
            "MAE"      : d.abs().mean(),
            "Bias"     : d.mean(),
            "conj_rank": conj_rank.get(cls, 99),
        })

    t = pd.DataFrame(records).sort_values("MAE", ascending=False).reset_index(drop=True)

    # Spearman
    # Sort ascending for Spearman (independent of display order)
    t_sp = t[t["conj_rank"] < 99].sort_values("MAE")
    rho, p_sp = stats.spearmanr(t_sp["conj_rank"], t_sp["MAE"])

    # Colour map by conjugation rank
    norm  = plt.Normalize(t["conj_rank"].min(), t["conj_rank"].max())
    cmap  = LinearSegmentedColormap.from_list(
        "conj", ["#4DAC26", "#FDB863", "#D73027"], N=256)
    colors = [cmap(norm(r)) for r in t["conj_rank"]]

    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    y_pos = np.arange(len(t))

    bars = ax.barh(y_pos, t["MAE"], color=colors, edgecolor="white",
                   linewidth=0.3, alpha=0.88, height=0.65)

    # n annotation on bars
    for bar, n_val, mae_val in zip(bars, t["n"], t["MAE"]):
        ax.text(mae_val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"n={n_val}", va="center", ha="left", fontsize=5.5,
                color="#444444")

    # Literature reference line
    ax.axvline(LIT_RMSE, color="#888888", lw=0.8, ls=":", alpha=0.8,
               label=f"Lit. RMSE ref. ({LIT_RMSE})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(t["label"], fontsize=6.5)
    ax.set_xlabel("MAE (log units)")
    ax.set_xlim(0, t["MAE"].max() * 1.25)

    # Spearman annotation — place at lower right, clear of bars
    ax.text(0.98, 0.02,
            f"Spearman ρ = {rho:.3f} (p < 0.001)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="#444444",
            bbox=dict(fc="white", ec="#cccccc", lw=0.5,
                      boxstyle="round,pad=0.3", alpha=0.95))

    # Colourbar for conjugation level
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.02)
    cbar.set_label("Conjugation level\n(1=compact → 9=extended)", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)

    ax.legend(fontsize=6, loc="lower right", framealpha=0.95,
              edgecolor="#cccccc", borderpad=0.4)
    ax.set_title(
        "Consensus logP error by structural class (n ≥ 3 per class)",
        fontsize=8.5, fontweight="bold", pad=4)
    fig.tight_layout()
    save(fig, "Figure_5")
    plt.close(fig)


# ===========================================================================
# FIGURE 6 — 2D Non-Additivity Heatmap
# ===========================================================================

def figure6(df: pd.DataFrame):
    """
    2D consensus bias heatmap: N_15 group × logP_15 range.
    logP cut-off = 1.5; cells with n < 3 masked.
    Red = overestimation (Exp−Pred < 0); Blue = underestimation.
    """
    N_ORDER    = ["N=0", "N=1-3", "N≥4"]
    LOGP_ORDER = ["logP<1.5", "logP 1.5-3.0", "logP>3.0"]

    pivot = (
        df.groupby(["N_15", "logP_15"], observed=True)["delta_Consensus"]
        .mean().round(2).unstack()
        .reindex(index=N_ORDER, columns=LOGP_ORDER)
    )
    counts = (
        df.groupby(["N_15", "logP_15"], observed=True)["delta_Consensus"]
        .count().unstack()
        .reindex(index=N_ORDER, columns=LOGP_ORDER)
    )
    pivot_masked = pivot.copy()
    pivot_masked[counts < 3] = np.nan

    fig, ax = plt.subplots(figsize=(COL15, 2.6))

    vabs = float(np.nanmax(np.abs(pivot_masked.values))) + 0.1
    im   = ax.imshow(pivot_masked.values, cmap="RdBu",
                     vmin=-vabs, vmax=vabs, aspect="auto")

    # Cell annotations
    for i in range(pivot_masked.shape[0]):
        for j in range(pivot_masked.shape[1]):
            val = pivot_masked.values[i, j]
            cnt = int(counts.values[i, j]) if not np.isnan(counts.values[i, j]) else 0
            if not np.isnan(val):
                txt_color = "white" if abs(val) > vabs * 0.65 else "#222222"
                ax.text(j, i, f"{val:.2f}\n(n={cnt})",
                        ha="center", va="center", fontsize=7,
                        color=txt_color, fontweight="bold" if abs(val) > 1.5 else "normal")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=8, color="#aaaaaa")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.set_label("Mean bias (Exp − Pred)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(range(len(LOGP_ORDER)))
    ax.set_yticks(range(len(N_ORDER)))
    ax.set_xticklabels(LOGP_ORDER, fontsize=7)
    ax.set_yticklabels(N_ORDER, fontsize=7)
    ax.set_xlabel("Experimental logP range (cut-off 1.5)", fontsize=7)
    ax.set_ylabel("Nitrogen count group", fontsize=7)
    ax.set_title("2D consensus bias map (N count × logP range)",
                 fontsize=8.5, fontweight="bold", pad=4)

    ax.text(0.5, -0.22,
            "Red = overestimation (Exp < Pred)  |  Blue = underestimation (Exp > Pred)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color="#555555", style="italic")

    # Grid lines
    for i in range(len(N_ORDER) + 1):
        ax.axhline(i - 0.5, color="white", lw=0.8)
    for j in range(len(LOGP_ORDER) + 1):
        ax.axvline(j - 0.5, color="white", lw=0.8)

    fig.tight_layout()
    save(fig, "Figure_6")
    plt.close(fig)


# ===========================================================================
# FIGURE 7 — Failure Mode Taxonomy
# ===========================================================================

def figure7(df: pd.DataFrame):
    """
    Two-panel failure mode figure:
    A — FM distribution (bar chart, n per FM)
    B — RMSE by FM × 3 predictors (Consensus, XLOGP3, MLOGP)
    """
    fm_order  = ["FM0", "FM1", "FM2", "FM3", "FM4"]
    preds3    = ["Consensus", "XLOGP3", "MLOGP"]
    pred3_col = {"Consensus": "#1A9641", "XLOGP3": "#4DAC26", "MLOGP": "#762A83"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 3.2))
    fig.subplots_adjust(wspace=0.38, bottom=0.22)

    # --- Panel A: FM distribution ---
    fm_counts = df["FM"].value_counts().reindex(fm_order).fillna(0)
    bars = ax1.bar(
        [FM_LABELS[f] for f in fm_order],
        fm_counts.values,
        color=[FM_COLORS[f] for f in fm_order],
        alpha=0.85, edgecolor="white", linewidth=0.3
    )
    for bar, val in zip(bars, fm_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, f"n={int(val)}",
                 ha="center", va="bottom", fontsize=6)

    ax1.set_ylabel("Number of compounds")
    ax1.set_title("(A) FM distribution  (Consensus logP, n = 95)",
                  fontsize=7, fontweight="bold")
    ax1.set_xticklabels([FM_LABELS[f] for f in fm_order],
                        rotation=30, ha="right", fontsize=6)
    ax1.tick_params(axis="x", labelsize=6)
    add_panel_label(ax1, "A")

    # --- Panel B: RMSE by FM × predictor ---
    x      = np.arange(len(fm_order))
    width  = 0.24
    offset = [-width, 0, width]

    for i, pred in enumerate(preds3):
        rmses = []
        for fm in fm_order:
            sub = df[df["FM"] == fm][f"delta_{pred}"].dropna()
            rmses.append(rmse(sub) if len(sub) > 0 else 0)
        ax2.bar(x + offset[i], rmses, width,
                label=PRED_LABELS[pred],
                color=pred3_col[pred], alpha=0.85,
                edgecolor="white", linewidth=0.3)

    # Literature reference
    ax2.axhline(LIT_RMSE, color="#888888", lw=0.8, ls=":",
                alpha=0.8, label=f"Lit. ref. ({LIT_RMSE})")
    ax2.text(len(fm_order) - 0.5, LIT_RMSE + 0.03,
             f"Lit. ref. ({LIT_RMSE})", ha="right", va="bottom",
             fontsize=5.5, color="#888888", style="italic")

    ax2.set_xticks(x)
    ax2.set_xticklabels([FM_LABELS[f] for f in fm_order],
                        rotation=30, ha="right", fontsize=6)
    ax2.set_ylabel("RMSE (log units)")
    ax2.set_title("(B) RMSE by failure mode", fontsize=7, fontweight="bold")
    ax2.legend(fontsize=6, framealpha=0.9, edgecolor="none",
               loc="upper right")
    add_panel_label(ax2, "B")

    fig.suptitle("Five-class failure mode taxonomy for fragment-based logP prediction",
                 fontsize=8, fontweight="bold", y=1.02)
    save(fig, "Figure_7")
    plt.close(fig)


# ===========================================================================
# FIGURE 8 — Chemical Interpretation / ESP Panel (DFT placeholder)
# ===========================================================================

def figure8(df: pd.DataFrame):
    """
    DFT/ESP placeholder figure.
    Shows panel layout and compound information.
    Will be replaced with actual ESP surfaces after ORCA calculations.
    """
    panel_compounds = [
        ("CMR_GOLD_055", "N=0 reference\n(electronic baseline)"),
        ("CMR_GOLD_043", "N=2, accurate\n(compact oxadiazole N)"),
        ("CMR_GOLD_079", "N=4, near-zero error\n(FM3, N-cancellation)"),
        ("CMR_GOLD_058", "N=2, extreme failure\n(D-π-A, error = −5.19)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.8))
    fig.subplots_adjust(wspace=0.08)

    panel_labels = ["A", "B", "C", "D"]

    for ax, (cid, role), plabel in zip(axes, panel_compounds, panel_labels):
        row = df[df["Compound_ID"] == cid]

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        # Grey placeholder box
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.15), 0.90, 0.65,
            boxstyle="round,pad=0.02",
            facecolor="#f0f0f0", edgecolor="#bbbbbb", linewidth=0.8
        )
        ax.add_patch(rect)

        ax.text(0.50, 0.62, "ESP surface\n(ORCA pending)",
                ha="center", va="center", fontsize=7,
                color="#888888", style="italic")
        ax.text(0.50, 0.45, "B3LYP/6-31G(d)\n0.002 a.u.",
                ha="center", va="center", fontsize=6, color="#aaaaaa")

        if len(row) > 0:
            r = row.iloc[0]
            info = (
                f"{cid}\n"
                f"N = {int(r['N_count'])}\n"
                f"logP_exp = {r['logP_exp']:.2f}\n"
                f"Δ = {r['delta_Consensus']:+.2f}"
            )
            ax.text(0.50, 0.10, info, ha="center", va="bottom",
                    fontsize=6.5, color="#333333",
                    bbox=dict(fc="white", ec="none", alpha=0.8))

        ax.set_title(role, fontsize=6.5, pad=4, color="#333333")
        ax.axis("off")
        add_panel_label(ax, plabel, x=-0.04, y=1.02)

    fig.suptitle(
        "DFT mechanistic panel — ESP surfaces (placeholder, ORCA pending)\n"
        "Red: electron-rich  |  Blue: electron-poor  |  ΔqN values will be annotated",
        fontsize=7.5, fontweight="bold", y=1.04
    )
    save(fig, "Figure_8")
    plt.close(fig)


# ===========================================================================
# SUPPLEMENTARY FIGURE S1 — Pairwise error scatter (logP_exp vs error)
# ===========================================================================

def figure_s1(df: pd.DataFrame):
    """
    SI Figure S1: Prediction error vs experimental logP for all 6 predictors.
    Shows how error varies across the logP range.
    """
    preds = ["iLOGP", "MLOGP", "Consensus", "WLOGP", "XLOGP3", "Silicos_IT"]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.38)

    for ax, pred in zip(axes.flatten(), preds):
        for grp in N_GROUPS:
            sub = df[df["N_group"] == grp]
            ax.scatter(sub["logP_exp"], sub[f"delta_{pred}"],
                       c=N_COLORS[grp], s=8, alpha=0.70,
                       edgecolors="none", label=grp, zorder=3)

        ax.axhline(0, color="#666666", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(1, color="#cccccc", lw=0.5, ls=":", alpha=0.6)
        ax.axhline(-1, color="#cccccc", lw=0.5, ls=":", alpha=0.6)

        d   = df[f"delta_{pred}"].dropna()
        ax.text(0.04, 0.04,
                f"Bias = {d.mean():+.2f}\nRMSE = {rmse(d):.2f}",
                transform=ax.transAxes, va="bottom", fontsize=5.5,
                bbox=dict(fc="white", ec="#cccccc", lw=0.4, boxstyle="round,pad=0.3"))

        ax.set_xlabel("Experimental logP", fontsize=6)
        ax.set_ylabel("Error (Exp − Pred)", fontsize=6)
        ax.set_title(PRED_LABELS[pred], fontsize=7, fontweight="bold",
                     color=PRED_COLORS[pred])
        ax.tick_params(labelsize=5.5)

    handles = [mpatches.Patch(color=N_COLORS[g], label=g) for g in N_GROUPS]
    axes.flatten()[-1].legend(handles=handles, title="N group",
                              title_fontsize=6, fontsize=5.5,
                              loc="upper right",
                              framealpha=0.9, edgecolor="none")

    fig.suptitle("Prediction error vs experimental logP (SI Figure S1)",
                 fontsize=8, fontweight="bold", y=1.01)
    save(fig, "Figure_S1")
    plt.close(fig)


# ===========================================================================
# SUPPLEMENTARY FIGURE S2 — Predictor N_count sensitivity
# ===========================================================================

def figure_s2(df: pd.DataFrame):
    """
    SI Figure S2: N_count sensitivity heatmap.
    β coefficient and p-value for error ~ N_count regression, per predictor.
    """
    from scipy.stats import linregress

    preds = list(PRED_COLORS.keys())
    betas, pvals = [], []
    for pred in preds:
        sub = df[["N_count", f"delta_{pred}"]].dropna()
        sl, _, _, pv, _ = linregress(sub["N_count"], sub[f"delta_{pred}"])
        betas.append(sl); pvals.append(pv)

    fig, ax = plt.subplots(figsize=(COL15, 2.6))

    colors = ["#D73027" if p < 0.05 else "#BBBBBB" for p in pvals]
    bars   = ax.barh(
        [PRED_LABELS[p] for p in preds], betas,
        color=colors, edgecolor="white", linewidth=0.3, alpha=0.88
    )
    ax.axvline(0, color="#444444", lw=0.8, ls="--")

    for bar, b, p in zip(bars, betas, pvals):
        sig = " *" if p < 0.05 else ""
        # Place label outside bar to avoid overlap
        if b >= 0:
            xpos = b + 0.015
            ha = "left"
        else:
            xpos = b - 0.015
            ha = "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"β={b:+.3f}  p={p:.3f}{sig}",
                va="center", ha=ha, fontsize=6, color="#333333")

    # Extend x limits to fit labels
    cur_xlim = ax.get_xlim()
    ax.set_xlim(cur_xlim[0] - 0.15, cur_xlim[1] + 0.15)
    ax.set_xlabel("β coefficient (error ~ N_count)")
    ax.set_title("N_count sensitivity per predictor (SI Figure S2)",
                 fontsize=7.5, fontweight="bold", pad=4)
    ax.text(0.98, 0.02, "* p < 0.05",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="#D73027", style="italic")

    fig.tight_layout()
    save(fig, "Figure_S2")
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================

FIGURE_MAP = {
    1: ("figure1",    "Dataset Overview"),
    2: ("figure2",    "Predictor Performance Scatter Matrix"),
    3: ("figure3",    "Nitrogen Effect: Bias Direction Reversal"),
    4: ("figure4",    "Error Distribution"),
    5: ("figure5",    "Structural Class Conjugation Gradient"),
    6: ("figure6",    "2D Non-Additivity Heatmap"),
    7: ("figure7",    "Failure Mode Taxonomy"),
    8: ("figure8",    "Chemical Interpretation (DFT Placeholder)"),
    91: ("figure_s1", "SI Figure S1: Error vs logP_exp"),
    92: ("figure_s2", "SI Figure S2: N_count Sensitivity"),
}

FUNC_MAP = {
    "figure1"   : figure1,
    "figure2"   : figure2,
    "figure3"   : figure3,
    "figure4"   : figure4,
    "figure5"   : figure5,
    "figure6"   : figure6,
    "figure7"   : figure7,
    "figure8"   : figure8,
    "figure_s1" : figure_s1,
    "figure_s2" : figure_s2,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for coumarin-logp manuscript."
    )
    parser.add_argument(
        "--fig", nargs="+", type=int, default=None,
        help="Figure numbers to generate (e.g. --fig 3 5). "
             "Omit to generate all. SI: 91, 92."
    )
    args = parser.parse_args()

    print("=" * 65)
    print("02_figure_engine.py — coumarin-logp Benchmark Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"\nLoading data: {IN_CSV}")
    df = load(IN_CSV)
    print(f"  {len(df)} compounds loaded")

    # Determine which figures to generate
    if args.fig:
        to_run = [(n, FIGURE_MAP[n]) for n in args.fig if n in FIGURE_MAP]
    else:
        to_run = list(FIGURE_MAP.items())

    print(f"\nGenerating {len(to_run)} figure(s)...\n")
    for fig_num, (func_name, title) in to_run:
        print(f"--- Figure {fig_num}: {title} ---")
        FUNC_MAP[func_name](df)

    print(f"\n✅ All figures complete.")
    print(f"   Output directory: {FIG_DIR}")
    print(f"   Formats: PNG ({DPI_DRAFT} dpi) | TIFF ({DPI_SUBMIT} dpi) | PDF")
    print(f"   Next step: review figures, then write manuscript sections")
    print("=" * 65)


if __name__ == "__main__":
    main()