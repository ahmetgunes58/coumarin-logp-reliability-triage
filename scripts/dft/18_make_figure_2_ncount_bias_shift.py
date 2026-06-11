from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, mannwhitneyu, kruskal

# ============================================================
# CONFIG
# ============================================================
PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

# Eğer otomatik bulma çalışmazsa, aşağıdaki satırı bir dosyaya sabitleyebilirsin:
# INPUT_CSV = PROJECT_DIR / "data" / "processed" / "YOUR_MASTER_DATASET.csv"
INPUT_CSV = None

OUT_DIR = PROJECT_DIR / "figures" / "manuscript"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT  = OUT_DIR / "Figure_2_ncount_bias_shift.png"
PDF_OUT  = OUT_DIR / "Figure_2_ncount_bias_shift.pdf"
TIFF_OUT = OUT_DIR / "Figure_2_ncount_bias_shift.tiff"
TXT_OUT  = OUT_DIR / "Figure_2_ncount_bias_shift_stats.txt"

# ============================================================
# HELPERS
# ============================================================
def norm_col(name: str) -> str:
    """Normalize column names for robust matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

def build_norm_map(columns):
    return {norm_col(c): c for c in columns}

def find_first_existing_col(norm_map, candidates):
    for cand in candidates:
        key = norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None

def find_input_csv(project_dir: Path) -> Path:
    """
    Search likely CSV files and return the first one containing:
    experimental logP, N_count, and six SwissADME predictor columns.
    """
    search_dirs = [
        project_dir / "data" / "processed",
        project_dir / "data",
        project_dir
    ]

    # aliases
    exp_candidates = [
        "logP_exp", "logp_exp", "experimental_logP", "experimental_logp",
        "exp_logP", "exp_logp", "logP experimental", "logPexp"
    ]
    n_candidates = [
        "N_count", "n_count", "nitrogen_count", "nitrogencount", "ncount"
    ]
    predictor_aliases = {
        "iLOGP":      ["iLOGP", "ilogp", "i_logp"],
        "XLOGP3":     ["XLOGP3", "xlogp3", "xlogp_3"],
        "WLOGP":      ["WLOGP", "wlogp"],
        "MLOGP":      ["MLOGP", "mlogp"],
        "Silicos-IT": ["Silicos-IT", "silicos_it", "silicosit", "silicos"],
        "Consensus":  ["Consensus", "consensus", "consensus_logP", "consensus_logp", "swissadme_consensus"]
    }

    candidate_files = []
    for d in search_dirs:
        if d.exists():
            candidate_files.extend(sorted(d.glob("*.csv")))

    for csv_file in candidate_files:
        try:
            df0 = pd.read_csv(csv_file, nrows=5)
        except Exception:
            continue

        nmap = build_norm_map(df0.columns)
        exp_col = find_first_existing_col(nmap, exp_candidates)
        n_col   = find_first_existing_col(nmap, n_candidates)

        if exp_col is None or n_col is None:
            continue

        ok = True
        for _, aliases in predictor_aliases.items():
            if find_first_existing_col(nmap, aliases) is None:
                ok = False
                break

        if ok:
            return csv_file

    raise FileNotFoundError(
        "Uygun input CSV otomatik bulunamadı. "
        "Script başındaki INPUT_CSV değişkenine veri dosyanın tam yolunu yaz."
    )

def get_column_mapping(df: pd.DataFrame):
    nmap = build_norm_map(df.columns)

    exp_col = find_first_existing_col(nmap, [
        "logP_exp", "logp_exp", "experimental_logP", "experimental_logp",
        "exp_logP", "exp_logp", "logP experimental", "logPexp"
    ])
    n_col = find_first_existing_col(nmap, [
        "N_count", "n_count", "nitrogen_count", "nitrogencount", "ncount"
    ])

    predictors = {
        "iLOGP":      find_first_existing_col(nmap, ["iLOGP", "ilogp", "i_logp"]),
        "XLOGP3":     find_first_existing_col(nmap, ["XLOGP3", "xlogp3", "xlogp_3"]),
        "WLOGP":      find_first_existing_col(nmap, ["WLOGP", "wlogp"]),
        "MLOGP":      find_first_existing_col(nmap, ["MLOGP", "mlogp"]),
        "Silicos-IT": find_first_existing_col(nmap, ["Silicos-IT", "silicos_it", "silicosit", "silicos"]),
        "Consensus":  find_first_existing_col(nmap, ["Consensus", "consensus", "consensus_logP", "consensus_logp", "swissadme_consensus"])
    }

    missing = []
    if exp_col is None:
        missing.append("experimental logP column")
    if n_col is None:
        missing.append("N_count column")
    for k, v in predictors.items():
        if v is None:
            missing.append(k)

    if missing:
        raise ValueError("Eksik sütunlar bulundu: " + ", ".join(missing))

    return exp_col, n_col, predictors

def n_group(n):
    if pd.isna(n):
        return np.nan
    n = int(n)
    if n == 0:
        return "N = 0"
    elif n == 1:
        return "N = 1"
    elif n in [2, 3]:
        return "N = 2–3"
    else:
        return "N ≥ 4"

def mean_ci(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    mean = x.mean()
    if n == 1:
        return mean, 0.0
    sem = x.sem(ddof=1)
    ci = t.ppf(0.975, df=n-1) * sem
    return mean, ci

# ============================================================
# LOAD DATA
# ============================================================
if INPUT_CSV is None:
    INPUT_CSV = find_input_csv(PROJECT_DIR)

df = pd.read_csv(INPUT_CSV)
exp_col, n_col, predictor_cols = get_column_mapping(df)

# keep only needed rows
keep_cols = [exp_col, n_col] + list(predictor_cols.values())
plot_df = df[keep_cols].copy()

plot_df[exp_col] = pd.to_numeric(plot_df[exp_col], errors="coerce")
plot_df[n_col]   = pd.to_numeric(plot_df[n_col], errors="coerce")

for c in predictor_cols.values():
    plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

plot_df = plot_df.dropna(subset=[exp_col, n_col]).copy()
plot_df["N_group"] = plot_df[n_col].apply(n_group)

group_order = ["N = 0", "N = 1", "N = 2–3", "N ≥ 4"]
plot_df["N_group"] = pd.Categorical(plot_df["N_group"], categories=group_order, ordered=True)

# ============================================================
# LONG FORMAT + SUMMARY
# ============================================================
records = []
for predictor_name, predictor_col in predictor_cols.items():
    tmp = plot_df[[exp_col, "N_group", predictor_col]].dropna().copy()
    tmp["Predictor"] = predictor_name
    tmp["Delta_logP"] = tmp[exp_col] - tmp[predictor_col]   # Exp - Pred
    records.append(tmp[["N_group", "Predictor", "Delta_logP"]])

long_df = pd.concat(records, ignore_index=True)

summary_rows = []
for g in group_order:
    for p in predictor_cols.keys():
        sub = long_df[(long_df["N_group"] == g) & (long_df["Predictor"] == p)]
        mean, ci = mean_ci(sub["Delta_logP"])
        summary_rows.append({
            "N_group": g,
            "Predictor": p,
            "mean": mean,
            "ci95": ci,
            "n": len(sub)
        })

summary_df = pd.DataFrame(summary_rows)

# group counts from original data
group_counts = plot_df["N_group"].value_counts().reindex(group_order)

# ============================================================
# OPTIONAL STATS REPORT (CONSENSUS ONLY)
# ============================================================
cons_df = long_df[long_df["Predictor"] == "Consensus"].copy()

n0 = cons_df.loc[cons_df["N_group"] == "N = 0", "Delta_logP"].dropna()
nge1 = cons_df.loc[cons_df["N_group"] != "N = 0", "Delta_logP"].dropna()

kw_groups = [
    cons_df.loc[cons_df["N_group"] == g, "Delta_logP"].dropna()
    for g in group_order
]

mw_p = np.nan
if len(n0) > 0 and len(nge1) > 0:
    mw_p = mannwhitneyu(n0, nge1, alternative="two-sided").pvalue

kw_p = np.nan
if all(len(x) > 0 for x in kw_groups):
    kw_p = kruskal(*kw_groups).pvalue

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("Figure 2 statistical summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Input file: {INPUT_CSV}\n\n")

    f.write("Group counts (from original dataset)\n")
    f.write("-" * 60 + "\n")
    for g in group_order:
        f.write(f"{g}: n = {int(group_counts[g])}\n")

    f.write("\nConsensus-only nonparametric tests\n")
    f.write("-" * 60 + "\n")
    f.write(f"Mann–Whitney (N = 0 vs N ≥ 1): p = {mw_p:.6g}\n")
    f.write(f"Kruskal–Wallis (all N groups): p = {kw_p:.6g}\n")

    f.write("\nMean ± 95% CI by group and predictor\n")
    f.write("-" * 60 + "\n")
    for _, row in summary_df.iterrows():
        f.write(
            f"{row['N_group']:>7} | {row['Predictor']:<10} | "
            f"mean = {row['mean']:+.3f} | CI95 = ±{row['ci95']:.3f} | n = {int(row['n'])}\n"
        )

# ============================================================
# PLOT
# ============================================================
predictor_order = ["iLOGP", "XLOGP3", "WLOGP", "MLOGP", "Silicos-IT", "Consensus"]

colors = {
    "iLOGP":      "#4C78A8",
    "XLOGP3":     "#54A24B",
    "WLOGP":      "#E457B9",
    "MLOGP":      "#9C6ADE",
    "Silicos-IT": "#F28E2B",
    "Consensus":  "#2F8F6B",
}

fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=300)

x = np.arange(len(group_order))
width = 0.12
offsets = (np.arange(len(predictor_order)) - (len(predictor_order) - 1) / 2) * width

for i, predictor in enumerate(predictor_order):
    sub = summary_df[summary_df["Predictor"] == predictor].set_index("N_group").loc[group_order]
    means = sub["mean"].values
    cis = sub["ci95"].values

    ax.bar(
        x + offsets[i],
        means,
        width=width,
        color=colors[predictor],
        edgecolor="black",
        linewidth=0.6,
        label=predictor,
        zorder=3
    )

    ax.errorbar(
        x + offsets[i],
        means,
        yerr=cis,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.8,
        capthick=0.8,
        zorder=4
    )

# zero line
ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, zorder=2)

# style
ax.set_axisbelow(True)
ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

# labels
xticklabels = [f"{g}\n(n = {int(group_counts[g])})" for g in group_order]
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=12)
ax.set_xlabel("Nitrogen-count group", fontsize=14)
ax.set_ylabel(r"Mean $\Delta$logP (Exp − Pred, log units)", fontsize=14)

# y limits with margin
ymin = np.nanmin(summary_df["mean"] - summary_df["ci95"])
ymax = np.nanmax(summary_df["mean"] + summary_df["ci95"])
pad = 0.35
ax.set_ylim(ymin - pad, ymax + pad)

# legend
leg = ax.legend(
    title="Predictor",
    ncol=1,
    frameon=False,
    fontsize=11,
    title_fontsize=12,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0)
)

# no in-figure title (cleaner for manuscript)
plt.tight_layout()

# save
fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight")
fig.savefig(PDF_OUT, bbox_inches="tight")
fig.savefig(TIFF_OUT, dpi=600, bbox_inches="tight")
plt.close(fig)

print("Figure 2 generated successfully.")
print(f"Input file : {INPUT_CSV}")
print(f"PNG output : {PNG_OUT}")
print(f"PDF output : {PDF_OUT}")
print(f"TIFF output: {TIFF_OUT}")
print(f"Stats file : {TXT_OUT}")