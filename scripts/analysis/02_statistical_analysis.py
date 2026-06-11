# =============================================================================
# 02_statistical_analysis.py
# coumarin-logp Benchmark Pipeline — Step 2: Statistical Analysis
#
# Author  : Ahmet GÜNEŞ
# Affil.  : National Defence University, Turkish Naval Academy,
#           Department of Basic Sciences, Istanbul, Türkiye
# Contact : ahmet.gunes3@msu.edu.tr
# Version : 2.0 (fully reproducible, RDKit-based N_count)
# Date    : 2026
#
# Description:
#   Runs all statistical analyses for the manuscript and produces
#   publication-ready CSV tables. Every number in the manuscript
#   originates from this script — no manual entry.
#
# Inputs:
#   data/processed/benchmark_dataset.csv   (from 01_prepare_dataset.py)
#
# Outputs:
#   data/processed/table1_overall_performance.csv
#   data/processed/table2_nitrogen_bias.csv
#   data/processed/table3_structural_class.csv
#   data/processed/table4_logp_range.csv
#   data/processed/table4b_regression.csv
#   data/processed/table4c_2d_heatmap_bias.csv
#   data/processed/table4c_2d_heatmap_counts.csv
#   data/processed/statistical_tests.txt
#   data/processed/manuscript_numbers.txt   ← all in-text numbers, one place
#
# Usage:
#   cd <project_root>
#   python scripts/02_statistical_analysis.py
#
# Requirements:
#   pandas >= 1.5  |  numpy >= 1.23  |  scipy >= 1.9
#
# Reproducibility:
#   No random operations. All tests are deterministic.
#   Results are identical across platforms given the same input CSV.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")
IN_CSV    = os.path.join(PROC_DIR, "benchmark_dataset.csv")

LIT_RMSE  = 0.60   # Mannhold et al. J. Pharm. Sci. 98 (2009) 861-893

PREDICTORS = ["iLOGP", "XLOGP3", "WLOGP", "MLOGP", "Silicos_IT", "Consensus"]
N_GROUPS   = ["N=0", "N=1", "N=2-3", "N≥4"]
LOGP_RANGES = ["logP<1", "logP 1-2", "logP 2-3", "logP>3"]

PARAM_LABELS = {
    "iLOGP"     : "Free energy perturbation",
    "XLOGP3"    : "Fragment + correction",
    "WLOGP"     : "Atomic contributions",
    "MLOGP"     : "Topological",
    "Silicos_IT": "Fragment-based",
    "Consensus" : "Ensemble (weighted)",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def rmse(series: pd.Series) -> float:
    return float(np.sqrt((series.dropna() ** 2).mean()))


def r_squared(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    R² = 1 - SS_res / SS_tot
    Can be negative when predictions are worse than the mean baseline.
    """
    mask = y_true.notna() & y_pred.notna()
    y, yp = y_true[mask].values, y_pred[mask].values
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def out(path: str, df: pd.DataFrame, msg: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {msg} → {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Input not found: {path}\nRun 01_prepare_dataset.py first.")
    df = pd.read_csv(path)
    df["N_group"]    = pd.Categorical(df["N_group"],    categories=N_GROUPS,    ordered=True)
    df["logP_range"] = pd.Categorical(df["logP_range"], categories=LOGP_RANGES, ordered=True)
    print(f"  Loaded {len(df)} compounds × {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# TABLE 1 — Overall predictor performance
# ---------------------------------------------------------------------------

def table1(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each predictor: Bias, MAE, RMSE, R², Fold vs literature,
    N_count β (slope of error ~ N_count regression), p-value.
    Sorted by ascending RMSE.
    """
    records = []
    for pred in PREDICTORS:
        d = df[f"delta_{pred}"].dropna()
        bias = d.mean()
        mae  = d.abs().mean()
        rms  = rmse(d)
        r2   = r_squared(df["logP_exp"], df[pred])
        fold = rms / LIT_RMSE

        # N_count sensitivity: slope of error ~ N_count
        sub = df[["N_count", f"delta_{pred}"]].dropna()
        slope, intercept, r_val, p_val, se_val = stats.linregress(
            sub["N_count"], sub[f"delta_{pred}"]
        )

        records.append({
            "Predictor"      : pred,
            "Parameterisation": PARAM_LABELS[pred],
            "Bias"           : round(bias, 3),
            "MAE"            : round(mae,  3),
            "RMSE"           : round(rms,  3),
            "R2"             : round(r2,   3),
            "Fold_vs_lit"    : round(fold, 1),
            "N_count_beta"   : round(slope, 3),
            "N_count_p"      : round(p_val, 3),
            "N_count_sig"    : stars(p_val),
        })

    t = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
    out(os.path.join(PROC_DIR, "table1_overall_performance.csv"), t,
        "Table 1 — overall performance")
    print(t.to_string(index=False))
    return t


# ---------------------------------------------------------------------------
# TABLE 2 — Nitrogen effect: bias by N group
# ---------------------------------------------------------------------------

def table2(df: pd.DataFrame) -> tuple:
    """
    Mean signed bias per N group for each predictor.
    Includes Mann-Whitney (N=0 vs N>=1) and Kruskal-Wallis (all groups).
    """
    records = []
    for grp in N_GROUPS:
        sub = df[df["N_group"] == grp]
        row = {"N_group": grp, "n": len(sub)}
        for pred in PREDICTORS:
            row[pred] = round(sub[f"delta_{pred}"].mean(), 2)
        records.append(row)

    t = pd.DataFrame(records)

    # Statistical tests on Consensus errors
    d0 = df[df["N_group"] == "N=0"]["delta_Consensus"].dropna()
    d1 = df[df["N_group"] != "N=0"]["delta_Consensus"].dropna()
    U, p_mw = stats.mannwhitneyu(d0, d1, alternative="two-sided")
    rb = float(1 - 2 * U / (len(d0) * len(d1)))  # rank-biserial r

    groups_kw = [
        df[df["N_group"] == g]["delta_Consensus"].dropna().values
        for g in N_GROUPS
    ]
    H, p_kw = stats.kruskal(*groups_kw)

    # N>=4 stratification by logP (for mechanistic discussion)
    n4 = df[df["N_group"] == "N≥4"]
    n4_low  = n4[n4["logP_exp"] < 2]
    n4_high = n4[n4["logP_exp"] >= 2]

    out(os.path.join(PROC_DIR, "table2_nitrogen_bias.csv"), t,
        "Table 2 — nitrogen bias")
    print(t.to_string(index=False))
    print(f"\n  Mann-Whitney (N=0 vs N≥1): U={U:.0f}, p={p_mw:.4f}, r={rb:.3f}")
    print(f"  Kruskal-Wallis (all groups): H={H:.2f}, p={p_kw:.4f}")
    print(f"  N≥4 stratification:")
    print(f"    logP<2:  n={len(n4_low)},  bias={n4_low['delta_Consensus'].mean():.2f}")
    print(f"    logP≥2:  n={len(n4_high)}, bias={n4_high['delta_Consensus'].mean():.2f}")

    return t, U, p_mw, rb, H, p_kw, n4_low, n4_high


# ---------------------------------------------------------------------------
# TABLE 3 — Structural class performance
# ---------------------------------------------------------------------------

def table3(df: pd.DataFrame) -> tuple:
    """
    Consensus logP performance by Coumarin_Type.
    Classes with n < 3 are excluded from the table but reported separately.
    Includes Spearman rho between MAE rank and conjugation level rank.
    """
    # Conjugation level rank (manually assigned, ascending conjugation)
    conj_rank = {
        "oxadiazole"                          : 1,
        "Furocoumarin"                        : 2,
        "oxadiazoline"                        : 3,
        "amidoxime"                           : 4,
        "simple_coumarin"                     : 5,
        "triazolo_thiadiazinyl_coumarin"      : 6,
        "conjugated_coumarin"                 : 7,
        "phosphonate_coumarin"                : 8,
        "dimeric_coumarin"                    : 9,
        "7-amino-substituted D-π-A coumarin"  : 10,
    }

    records = []
    small_classes = []
    for cls, sub in df.groupby("Coumarin_Type"):
        d = sub["delta_Consensus"].dropna()
        row = {
            "Structural_Class"  : cls,
            "n"                 : len(sub),
            "MAE"               : round(d.abs().mean(), 2),
            "RMSE"              : round(rmse(d), 2),
            "Bias"              : round(d.mean(), 2),
            "Conjugation_rank"  : conj_rank.get(cls, 99),
        }
        if len(sub) >= 3:
            records.append(row)
        else:
            small_classes.append(row)

    t = pd.DataFrame(records).sort_values("MAE").reset_index(drop=True)

    # Spearman rho: MAE rank vs conjugation rank
    t_sp = t[t["Conjugation_rank"] < 99].copy()
    rho, p_sp = stats.spearmanr(t_sp["Conjugation_rank"], t_sp["MAE"])

    # Classes excluded (n < 3)
    n_excluded = sum(r["n"] for r in small_classes)
    excluded_names = [r["Structural_Class"] for r in small_classes]

    out(os.path.join(PROC_DIR, "table3_structural_class.csv"), t,
        "Table 3 — structural class")
    print(t[["Structural_Class","n","MAE","RMSE","Bias"]].to_string(index=False))
    print(f"\n  Spearman rho (MAE vs conjugation rank): ρ={rho:.3f}, p={p_sp:.4f}")
    print(f"  Excluded (n<3): {len(small_classes)} classes, {n_excluded} compounds")
    print(f"  Excluded classes: {excluded_names}")

    return t, rho, p_sp, n_excluded, excluded_names


# ---------------------------------------------------------------------------
# TABLE 4 — logP range performance
# ---------------------------------------------------------------------------

def table4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consensus, XLOGP3, and MLOGP performance stratified by logP range.
    """
    records = []
    for rng in LOGP_RANGES:
        sub = df[df["logP_range"] == rng]
        if len(sub) == 0:
            continue
        row = {"logP_Range": rng, "n": len(sub)}
        for pred in ["Consensus", "XLOGP3", "MLOGP"]:
            d = sub[f"delta_{pred}"].dropna()
            row[f"{pred}_Bias"] = round(d.mean(), 2)
            row[f"{pred}_RMSE"] = round(rmse(d), 2)
        records.append(row)

    t = pd.DataFrame(records)
    out(os.path.join(PROC_DIR, "table4_logp_range.csv"), t,
        "Table 4 — logP range")
    print(t.to_string(index=False))
    return t


# ---------------------------------------------------------------------------
# TABLE 4b — Multiple regression: error ~ N_count + TPSA
# ---------------------------------------------------------------------------

def table4b(df: pd.DataFrame) -> tuple:
    """
    OLS regression of consensus prediction error on N_count and TPSA.
    Reports β, SE, t, p, 95% CI, VIF.
    Also reports univariate N_count regression for comparison.
    """
    sub = df[["delta_Consensus", "N_count", "TPSA"]].dropna()
    y = sub["delta_Consensus"].values
    X = np.column_stack([
        np.ones(len(sub)),
        sub["N_count"].values,
        sub["TPSA"].values,
    ])
    n_obs, k = len(y), X.shape[1]

    # OLS
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - k)
    mse    = ss_res / (n_obs - k)
    cov    = mse * np.linalg.inv(X.T @ X)
    se     = np.sqrt(np.diag(cov))
    t_stat = coeffs / se
    p_vals = np.array([
        2 * (1 - stats.t.cdf(abs(t), df=n_obs - k)) for t in t_stat
    ])
    # 95% CI
    t_crit = stats.t.ppf(0.975, df=n_obs - k)
    ci_lo  = coeffs - t_crit * se
    ci_hi  = coeffs + t_crit * se

    # VIF
    corr_X = np.corrcoef(X[:, 1:].T)
    vif    = np.diag(np.linalg.inv(corr_X))

    terms = ["Intercept", "N_count", "TPSA"]
    records = []
    for i, term in enumerate(terms):
        records.append({
            "Term"   : term,
            "beta"   : round(float(coeffs[i]), 4),
            "SE"     : round(float(se[i]), 4),
            "t"      : round(float(t_stat[i]), 2),
            "p"      : round(float(p_vals[i]), 4),
            "p_sig"  : stars(float(p_vals[i])),
            "CI_lo"  : round(float(ci_lo[i]), 3),
            "CI_hi"  : round(float(ci_hi[i]), 3),
            "VIF"    : round(float(vif[i - 1]), 2) if i > 0 else "—",
        })

    t_df = pd.DataFrame(records)

    # Univariate N_count regression
    sl_uni, int_uni, r_uni, p_uni, se_uni = stats.linregress(
        sub["N_count"], sub["delta_Consensus"]
    )
    adj_r2_uni = r_uni ** 2 - (1 - r_uni ** 2) / (n_obs - 2)

    out(os.path.join(PROC_DIR, "table4b_regression.csv"), t_df,
        "Table 4b — regression")
    print(t_df.to_string(index=False))
    print(f"\n  Adj-R² (multivariate) = {adj_r2:.4f}")
    print(f"  Univariate N_count: β={sl_uni:.4f}, p={p_uni:.4f}, adj-R²={adj_r2_uni:.4f}")

    return t_df, adj_r2, coeffs, p_vals, vif, sl_uni, p_uni, adj_r2_uni


# ---------------------------------------------------------------------------
# TABLE 4c — 2D bias heatmap (N group × logP range, cut-off 1.5)
# ---------------------------------------------------------------------------

def table4c(df: pd.DataFrame) -> tuple:
    """
    Cross-tabulation of consensus bias by N_15 group and logP_15 range.
    logP cut-off = 1.5 (scientifically motivated: Lipinski polar threshold).
    N=1 and N=2-3 are collapsed to N=1-3.
    Cells with n < 3 are masked as NaN.
    """
    N_ORDER    = ["N=0", "N=1-3", "N≥4"]
    LOGP_ORDER = ["logP<1.5", "logP 1.5-3.0", "logP>3.0"]

    pivot = (
        df.groupby(["N_15", "logP_15"], observed=True)["delta_Consensus"]
        .mean()
        .round(2)
        .unstack()
        .reindex(index=N_ORDER, columns=LOGP_ORDER)
    )
    counts = (
        df.groupby(["N_15", "logP_15"], observed=True)["delta_Consensus"]
        .count()
        .unstack()
        .reindex(index=N_ORDER, columns=LOGP_ORDER)
    )

    # Mask cells with n < 3
    pivot_masked = pivot.copy()
    pivot_masked[counts < 3] = np.nan

    out(os.path.join(PROC_DIR, "table4c_2d_heatmap_bias.csv"),
        pivot_masked.reset_index(), "Table 4c — 2D bias")
    out(os.path.join(PROC_DIR, "table4c_2d_heatmap_counts.csv"),
        counts.reset_index(), "Table 4c — 2D counts")

    print("\n  Bias:")
    print(pivot_masked.to_string())
    print("\n  Counts:")
    print(counts.to_string())

    return pivot_masked, counts


# ---------------------------------------------------------------------------
# STATISTICAL TESTS (comprehensive)
# ---------------------------------------------------------------------------

def statistical_tests(df: pd.DataFrame) -> dict:
    """
    All hypothesis tests reported in the manuscript.
    Returns a dict for use in manuscript_numbers.txt.
    """
    results = {}
    lines   = []
    lines.append("=" * 65)
    lines.append("STATISTICAL TESTS — coumarin-logp Benchmark (n=95)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    # 1. Mann-Whitney: N=0 vs N>=1
    d0 = df[df["N_group"] == "N=0"]["delta_Consensus"].dropna()
    d1 = df[df["N_group"] != "N=0"]["delta_Consensus"].dropna()
    U, p_mw = stats.mannwhitneyu(d0, d1, alternative="two-sided")
    rb = float(1 - 2 * U / (len(d0) * len(d1)))
    results["mw_U"]  = U
    results["mw_p"]  = round(p_mw, 4)
    results["mw_rb"] = round(rb, 3)
    lines.append(f"\n1. Mann-Whitney U (N=0 vs N≥1, Consensus)")
    lines.append(f"   U = {U:.0f}, p = {p_mw:.4f}, rank-biserial r = {rb:.3f}")

    # 2. Kruskal-Wallis: all N groups
    groups_kw = [
        df[df["N_group"] == g]["delta_Consensus"].dropna().values
        for g in N_GROUPS
    ]
    H, p_kw = stats.kruskal(*groups_kw)
    results["kw_H"] = round(H, 2)
    results["kw_p"] = round(p_kw, 4)
    lines.append(f"\n2. Kruskal-Wallis H (all N groups, Consensus)")
    lines.append(f"   H = {H:.2f}, p = {p_kw:.4f}")

    # 3. Shapiro-Wilk: error normality
    err   = df["delta_Consensus"].dropna()
    W, p_sw = stats.shapiro(err)
    results["sw_W"] = round(W, 3)
    results["sw_p"] = round(p_sw, 3)
    lines.append(f"\n3. Shapiro-Wilk (Consensus errors, n={len(err)})")
    lines.append(f"   W = {W:.3f}, p = {p_sw:.3f}")

    # 4. Error distribution
    pct_neg  = float((err < 0).mean() * 100)
    pct_abs1 = float((err.abs() > 1.0).mean() * 100)
    pct_abs2 = float((err.abs() > 2.0).mean() * 100)
    results["mean_bias"]   = round(float(err.mean()), 3)
    results["median_bias"] = round(float(err.median()), 3)
    results["pct_overest"] = round(pct_neg, 1)
    results["pct_abs1"]    = round(pct_abs1, 1)
    results["pct_abs2"]    = round(pct_abs2, 1)
    lines.append(f"\n4. Error distribution (Consensus)")
    lines.append(f"   Mean   = {err.mean():.3f}  |  Median = {err.median():.3f}")
    lines.append(f"   Overestimated (< 0): {pct_neg:.1f}%")
    lines.append(f"   |error| > 1.0 log units: {pct_abs1:.1f}%")
    lines.append(f"   |error| > 2.0 log units: {pct_abs2:.1f}%")

    # 5. Spearman: N_count vs logP_exp
    rho_sp, p_sp = stats.spearmanr(df["N_count"], df["logP_exp"])
    results["spearman_rho"] = round(float(rho_sp), 3)
    results["spearman_p"]   = round(float(p_sp), 3)
    lines.append(f"\n5. Spearman (N_count vs logP_exp)")
    lines.append(f"   ρ = {rho_sp:.3f}, p = {p_sp:.3f}")

    # 6. Univariate regression: error ~ N_count
    sub = df[["delta_Consensus","N_count"]].dropna()
    sl, ic, r, p_uni, se = stats.linregress(sub["N_count"], sub["delta_Consensus"])
    n2 = len(sub)
    adj_r2_uni = r ** 2 - (1 - r ** 2) / (n2 - 2)
    results["uni_beta"]   = round(float(sl), 4)
    results["uni_p"]      = round(float(p_uni), 4)
    results["uni_adjr2"]  = round(float(adj_r2_uni), 4)
    lines.append(f"\n6. Univariate regression (error ~ N_count)")
    lines.append(f"   β = {sl:.4f}, p = {p_uni:.4f}, adj-R² = {adj_r2_uni:.4f}")

    # 7. Per-predictor N_count sensitivity
    lines.append(f"\n7. N_count sensitivity per predictor (error ~ N_count)")
    for pred in PREDICTORS:
        sub2 = df[["N_count", f"delta_{pred}"]].dropna()
        sl2, _, _, p2, _ = stats.linregress(sub2["N_count"], sub2[f"delta_{pred}"])
        lines.append(f"   {pred:12s}: β = {sl2:+.3f}, p = {p2:.3f} {stars(p2)}")

    # 8. N>=4 stratification by logP
    n4 = df[df["N_group"] == "N≥4"]
    n4l = n4[n4["logP_exp"] < 2]["delta_Consensus"]
    n4h = n4[n4["logP_exp"] >= 2]["delta_Consensus"]
    results["n4_low_n"]    = len(n4l)
    results["n4_low_bias"] = round(float(n4l.mean()), 2)
    results["n4_high_n"]   = len(n4h)
    results["n4_high_bias"]= round(float(n4h.mean()), 2)
    lines.append(f"\n8. N≥4 stratification by logP (Consensus)")
    lines.append(f"   logP < 2: n={len(n4l)}, bias={n4l.mean():.2f}")
    lines.append(f"   logP ≥ 2: n={len(n4h)}, bias={n4h.mean():.2f}")

    # 9. RMSE context
    rmse_range = {}
    for pred in PREDICTORS:
        rms = rmse(df[f"delta_{pred}"])
        fold = rms / LIT_RMSE
        rmse_range[pred] = {"rmse": round(rms, 3), "fold": round(fold, 1)}
    lines.append(f"\n9. RMSE vs literature benchmark ({LIT_RMSE} log units)")
    for pred, v in rmse_range.items():
        lines.append(f"   {pred:12s}: RMSE={v['rmse']:.3f}, fold={v['fold']:.1f}x")

    lines.append("\n" + "=" * 65)
    report = "\n".join(lines)
    print(report)

    path = os.path.join(PROC_DIR, "statistical_tests.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✅ Statistical tests → {os.path.basename(path)}")

    return results


# ---------------------------------------------------------------------------
# MANUSCRIPT NUMBERS — single source of truth
# ---------------------------------------------------------------------------

def manuscript_numbers(
    df, t1, t2_stats, t3_stats, t4, t4b_stats, t4c, stat_results
) -> None:
    """
    Writes every in-text number to a single file.
    When writing the manuscript, copy from here — never from memory.
    """
    U, p_mw, rb, H, p_kw, n4_low, n4_high = t2_stats
    t3, rho_conj, p_conj, n_excl, excl_names = t3_stats
    t4b_df, adj_r2, coeffs, p_vals, vif, sl_uni, p_uni, adj_r2_uni = t4b_stats
    pivot4c, counts4c = t4c
    sr = stat_results

    lines = []
    lines.append("=" * 65)
    lines.append("MANUSCRIPT NUMBERS — single source of truth")
    lines.append("Copy these values directly into the manuscript text.")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    lines.append("\n--- DATASET ---")
    lines.append(f"n = {len(df)}")
    lines.append(f"logP range: {df['logP_exp'].min():.2f}–{df['logP_exp'].max():.2f}")
    lines.append(f"logP mean (SD): {df['logP_exp'].mean():.2f} ({df['logP_exp'].std():.2f})")
    lines.append(f"GOLD: {sum(df['Data_Tier']=='GOLD')} | STRICT: {sum(df['Data_Tier']=='STRICT')} | EXTENDED: {sum(df['Data_Tier']=='EXTENDED')}")
    lines.append(f"N groups: {dict(df['N_group'].value_counts().sort_index())}")

    lines.append("\n--- TABLE 1 — RMSE range ---")
    rmse_min = t1["RMSE"].min(); rmse_max = t1["RMSE"].max()
    fold_min = t1["Fold_vs_lit"].min(); fold_max = t1["Fold_vs_lit"].max()
    lines.append(f"RMSE range: {rmse_min:.3f}–{rmse_max:.3f} log units")
    lines.append(f"Fold range: {fold_min:.1f}×–{fold_max:.1f}×")
    lines.append(f"All R² < 0: {(t1['R2'] < 0).all()}")
    lines.append(f"iLOGP: β_N = +{t1[t1.Predictor=='iLOGP']['N_count_beta'].values[0]:.3f} (p={t1[t1.Predictor=='iLOGP']['N_count_p'].values[0]:.3f})")
    lines.append(f"XLOGP3: β_N = {t1[t1.Predictor=='XLOGP3']['N_count_beta'].values[0]:.3f} (p={t1[t1.Predictor=='XLOGP3']['N_count_p'].values[0]:.3f})")

    lines.append("\n--- TABLE 2 — NITROGEN EFFECT ---")
    for grp in N_GROUPS:
        row = df[df["N_group"]==grp]
        bias = row["delta_Consensus"].mean()
        n = len(row)
        lines.append(f"  {grp} (n={n}): Consensus bias = {bias:+.2f}")
    lines.append(f"Mann-Whitney: U={U:.0f}, p={p_mw:.4f}")
    lines.append(f"Kruskal-Wallis: H={H:.2f}, p={p_kw:.4f}")
    lines.append(f"N≥4 stratification: logP<2 n={len(n4_low)} bias={n4_low['delta_Consensus'].mean():.2f} | logP≥2 n={len(n4_high)} bias={n4_high['delta_Consensus'].mean():.2f}")

    lines.append("\n--- ERROR DISTRIBUTION ---")
    lines.append(f"Mean bias = {sr['mean_bias']:.3f}")
    lines.append(f"Median bias = {sr['median_bias']:.3f}")
    lines.append(f"% overestimated = {sr['pct_overest']:.1f}%")
    lines.append(f"|error| > 1.0: {sr['pct_abs1']:.1f}%")
    lines.append(f"|error| > 2.0: {sr['pct_abs2']:.1f}%")
    lines.append(f"Shapiro-Wilk: W={sr['sw_W']:.3f}, p={sr['sw_p']:.3f}")

    lines.append("\n--- TABLE 3 — STRUCTURAL CLASS ---")
    lines.append(f"MAE range: {t3['MAE'].min():.2f}–{t3['MAE'].max():.2f} (fold={t3['MAE'].max()/t3['MAE'].min():.0f}x)")
    lines.append(f"Spearman rho (MAE vs conjugation): ρ={rho_conj:.3f}, p={p_conj:.4f}")
    lines.append(f"Excluded classes (n<3): {n_excl} compounds across {len(excl_names)} subtypes")
    for _, row in t3.iterrows():
        lines.append(f"  {row['Structural_Class']}: n={row['n']}, MAE={row['MAE']:.2f}, Bias={row['Bias']:+.2f}")

    lines.append("\n--- TABLE 4 — logP RANGE ---")
    for _, row in t4.iterrows():
        lines.append(f"  {row['logP_Range']} (n={row['n']}): Cons bias={row['Consensus_Bias']:+.2f}, RMSE={row['Consensus_RMSE']:.2f}")

    lines.append("\n--- TABLE 4b — REGRESSION ---")
    lines.append(f"β_intercept = {coeffs[0]:+.4f} (p={p_vals[0]:.4f})")
    lines.append(f"β_N_count   = {coeffs[1]:+.4f} (p={p_vals[1]:.4f})")
    lines.append(f"β_TPSA      = {coeffs[2]:+.4f} (p={p_vals[2]:.4f})")
    lines.append(f"Adj-R² (multivariate) = {adj_r2:.4f} ({adj_r2*100:.1f}%)")
    lines.append(f"Residual variance = {(1-adj_r2)*100:.1f}%")
    lines.append(f"VIF_N = {vif[0]:.2f}, VIF_TPSA = {vif[1]:.2f}")
    lines.append(f"Univariate N_count: β={sl_uni:.4f}, p={p_uni:.4f}, adj-R²={adj_r2_uni:.4f}")

    lines.append("\n--- TABLE 4c — 2D HEATMAP ---")
    lines.append("(logP cut-off = 1.5; cells with n<3 = NaN)")
    lines.append(pivot4c.to_string())
    lines.append("\nCounts:")
    lines.append(counts4c.to_string())

    lines.append("\n--- WORST / BEST ERRORS ---")
    worst3 = df.nsmallest(5, "delta_Consensus")[
        ["Compound_ID","logP_exp","Consensus","delta_Consensus","N_count","Coumarin_Type"]
    ]
    best3 = df.nlargest(3, "delta_Consensus")[
        ["Compound_ID","logP_exp","Consensus","delta_Consensus","N_count","Coumarin_Type"]
    ]
    lines.append("Largest overestimation errors (most negative):")
    lines.append(worst3.to_string(index=False))
    lines.append("\nLargest underestimation errors (most positive):")
    lines.append(best3.to_string(index=False))

    lines.append("\n--- FM DISTRIBUTION ---")
    for fm, n in df["FM"].value_counts().sort_index().items():
        lines.append(f"  {fm}: n={n}")

    lines.append("\n--- DFT PANEL COMPOUNDS ---")
    panel = ["CMR_GOLD_055","CMR_GOLD_043","CMR_GOLD_079","CMR_GOLD_058"]
    for cid in panel:
        row = df[df["Compound_ID"]==cid]
        if len(row) > 0:
            r = row.iloc[0]
            lines.append(
                f"  {cid}: N={r['N_count']}, logP_exp={r['logP_exp']:.3f}, "
                f"Cons={r['Consensus']:.2f}, error={r['delta_Consensus']:.2f}, "
                f"MW={r['MW']:.1f}, TPSA={r['TPSA']:.1f}"
            )

    lines.append("\n" + "=" * 65)
    report = "\n".join(lines)
    print(report)

    path = os.path.join(PROC_DIR, "manuscript_numbers.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✅ Manuscript numbers → {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("02_statistical_analysis.py — coumarin-logp Benchmark Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    os.makedirs(PROC_DIR, exist_ok=True)

    print("\nLoading data...")
    df = load(IN_CSV)

    print("\n--- TABLE 1: Overall Performance ---")
    t1 = table1(df)

    print("\n--- TABLE 2: Nitrogen Effect ---")
    U, p_mw, rb, H, p_kw, n4_low, n4_high = [None]*7
    t2, U, p_mw, rb, H, p_kw, n4_low, n4_high = table2(df)
    t2_stats = (U, p_mw, rb, H, p_kw, n4_low, n4_high)

    print("\n--- TABLE 3: Structural Class ---")
    t3, rho_conj, p_conj, n_excl, excl_names = table3(df)
    t3_stats = (t3, rho_conj, p_conj, n_excl, excl_names)

    print("\n--- TABLE 4: logP Range ---")
    t4 = table4(df)

    print("\n--- TABLE 4b: Regression ---")
    t4b_df, adj_r2, coeffs, p_vals, vif, sl_uni, p_uni, adj_r2_uni = table4b(df)
    t4b_stats = (t4b_df, adj_r2, coeffs, p_vals, vif, sl_uni, p_uni, adj_r2_uni)

    print("\n--- TABLE 4c: 2D Heatmap ---")
    pivot4c, counts4c = table4c(df)
    t4c_stats = (pivot4c, counts4c)

    print("\n--- STATISTICAL TESTS ---")
    stat_results = statistical_tests(df)

    print("\n--- MANUSCRIPT NUMBERS ---")
    manuscript_numbers(df, t1, t2_stats, t3_stats, t4, t4b_stats, t4c_stats, stat_results)

    print(f"\n✅ All analyses complete.")
    print(f"   Outputs in: {PROC_DIR}")
    print(f"   Next step : python scripts/02_figure_engine.py")
    print("=" * 65)


if __name__ == "__main__":
    main()