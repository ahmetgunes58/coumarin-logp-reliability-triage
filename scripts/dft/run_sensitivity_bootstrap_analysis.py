# run_sensitivity_bootstrap_analysis.py
# Sensitivity analysis + FM0-FM4 bootstrap robustness analysis
# For coumarin-logp coumarin logP manuscript

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")

# Try common dataset locations
CANDIDATE_FILES = [
    ROOT / "Dataset_S1_benchmark_dataset.csv",
    ROOT / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv",
    ROOT / "data" / "Dataset_S1_benchmark_dataset.csv",
]

OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

BOOT_N = 10000
RANDOM_SEED = 42


def find_dataset():
    for p in CANDIDATE_FILES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset_S1_benchmark_dataset.csv bulunamadı. "
        "Dosyayı proje ana klasörüne veya data/processed klasörüne koy."
    )


def find_col(df, candidates, required=True):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fuzzy search
    for col in df.columns:
        low = col.lower()
        for cand in candidates:
            if cand.lower() in low:
                return col
    if required:
        raise KeyError(
            f"Gerekli kolon bulunamadı. Adaylar: {candidates}\n"
            f"Mevcut kolonlar: {list(df.columns)}"
        )
    return None


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def rmse(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def mae(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x)))


def ci_bootstrap(values, stat_func=np.mean, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(stat_func(sample))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def safe_group_label(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper().replace("_", "-").replace(" ", "-")


def dataset_summary(df, delta_col, logp_col, n_col, tier_name):
    delta = to_numeric(df[delta_col])
    logp = to_numeric(df[logp_col])
    n_count = to_numeric(df[n_col])

    out = {}
    out["Dataset"] = tier_name
    out["n"] = int(len(df))
    out["Consensus_mean_bias"] = float(delta.mean())
    out["Consensus_median_bias"] = float(delta.median())
    out["Consensus_MAE"] = mae(delta)
    out["Consensus_RMSE"] = rmse(delta)
    out["Overestimated_n"] = int((delta < 0).sum())
    out["Overestimated_percent"] = float(100 * (delta < 0).mean())

    # N = 0 vs N-containing
    n0 = df[n_count == 0]
    npos = df[n_count > 0]
    out["N0_n"] = int(len(n0))
    out["N0_mean_bias"] = float(to_numeric(n0[delta_col]).mean()) if len(n0) else np.nan
    out["Npos_n"] = int(len(npos))
    out["Npos_mean_bias"] = float(to_numeric(npos[delta_col]).mean()) if len(npos) else np.nan
    out["Npos_minus_N0_bias_shift"] = (
        out["Npos_mean_bias"] - out["N0_mean_bias"]
        if len(n0) and len(npos)
        else np.nan
    )

    # polar logP <= 1.0
    polar = df[logp <= 1.0]
    out["Polar_logP_le_1_n"] = int(len(polar))
    out["Polar_logP_le_1_mean_bias"] = (
        float(to_numeric(polar[delta_col]).mean()) if len(polar) else np.nan
    )
    out["Polar_logP_le_1_RMSE"] = (
        rmse(to_numeric(polar[delta_col])) if len(polar) else np.nan
    )

    # principal high-risk cell: N = 1-3 and logP < 1.5
    highrisk = df[(n_count >= 1) & (n_count <= 3) & (logp < 1.5)]
    out["Highrisk_N1_3_logP_lt_1p5_n"] = int(len(highrisk))
    out["Highrisk_N1_3_logP_lt_1p5_mean_bias"] = (
        float(to_numeric(highrisk[delta_col]).mean()) if len(highrisk) else np.nan
    )
    out["Highrisk_N1_3_logP_lt_1p5_MAE"] = (
        mae(to_numeric(highrisk[delta_col])) if len(highrisk) else np.nan
    )

    return out


def fm_bootstrap(df, delta_col, fm_col):
    rows = []
    for mode in ["FM0", "FM1", "FM2", "FM3", "FM4"]:
        sub = df[df[fm_col].astype(str).str.upper().str.contains(mode, na=False)].copy()
        values = to_numeric(sub[delta_col]).dropna().values
        if len(values) == 0:
            continue

        mean_bias = float(np.mean(values))
        median_bias = float(np.median(values))
        mae_val = float(np.mean(np.abs(values)))
        rmse_val = float(np.sqrt(np.mean(values ** 2)))

        mean_ci = ci_bootstrap(values, np.mean, BOOT_N, RANDOM_SEED)
        median_ci = ci_bootstrap(values, np.median, BOOT_N, RANDOM_SEED + 1)
        mae_ci = ci_bootstrap(np.abs(values), np.mean, BOOT_N, RANDOM_SEED + 2)

        rows.append({
            "Mode": mode,
            "n": int(len(values)),
            "Mean_bias": mean_bias,
            "Mean_bias_CI95_low": mean_ci[0],
            "Mean_bias_CI95_high": mean_ci[1],
            "Median_bias": median_bias,
            "Median_bias_CI95_low": median_ci[0],
            "Median_bias_CI95_high": median_ci[1],
            "MAE": mae_val,
            "MAE_CI95_low": mae_ci[0],
            "MAE_CI95_high": mae_ci[1],
            "RMSE": rmse_val,
        })
    return pd.DataFrame(rows)


def main():
    dataset_path = find_dataset()
    print(f"Dataset bulundu: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Satır sayısı: {len(df)}")
    print("Kolonlar:")
    for c in df.columns:
        print(" -", c)

    # Auto-detect columns
    tier_col = find_col(df, ["Data_Tier", "Tier", "Documentation_Tier", "data_tier"])
    logp_col = find_col(df, ["logP_exp", "Exp_logP", "experimental_logP", "Experimental_logP"])
    n_col = find_col(df, ["N_count", "Nitrogen_count", "NCount", "N"])
    fm_col = find_col(df, ["FM_label", "Failure_mode", "FM", "Failure_Mode"])
    consensus_col = find_col(df, ["Consensus", "Consensus_logP", "logP_consensus"], required=False)

    delta_col = find_col(
        df,
        ["delta_Consensus", "Delta_Consensus", "Consensus_error", "error_Consensus", "ΔlogP"],
        required=False,
    )

    if delta_col is None:
        if consensus_col is None:
            raise KeyError(
                "Consensus error kolonu bulunamadı ve Consensus logP kolonu da yok. "
                "Dataset_S1 içinde delta_Consensus veya Consensus_logP olmalı."
            )
        df["delta_Consensus_auto"] = to_numeric(df[logp_col]) - to_numeric(df[consensus_col])
        delta_col = "delta_Consensus_auto"
        print("delta_Consensus bulunamadı; logP_exp - Consensus_logP olarak hesaplandı.")

    print("\nKullanılan kolonlar:")
    print(f"Tier        : {tier_col}")
    print(f"logP_exp    : {logp_col}")
    print(f"N_count     : {n_col}")
    print(f"FM label    : {fm_col}")
    print(f"Delta/error : {delta_col}")

    # Normalise tier
    df["_tier_norm"] = df[tier_col].apply(safe_group_label)

    full = df.copy()
    high_conf = df[df["_tier_norm"].isin(["STRICT-GOLD", "GOLD", "STRICT"])].copy()
    strict = df[df["_tier_norm"].isin(["STRICT-GOLD", "STRICT"])].copy()

    datasets = [
        ("Full dataset: STRICT-GOLD + GOLD + EXTENDED", full),
        ("High-confidence: STRICT-GOLD + GOLD", high_conf),
        ("STRICT-GOLD only", strict),
    ]

    sensitivity_rows = []
    for name, sub in datasets:
        sensitivity_rows.append(dataset_summary(sub, delta_col, logp_col, n_col, name))

    sensitivity = pd.DataFrame(sensitivity_rows)

    # Bootstrap on full dataset for FM0-FM4
    fm_boot = fm_bootstrap(full, delta_col, fm_col)

    # Optional: FM bootstrap in high-confidence dataset
    fm_boot_highconf = fm_bootstrap(high_conf, delta_col, fm_col)
    if not fm_boot_highconf.empty:
        fm_boot_highconf.insert(0, "Dataset", "STRICT-GOLD + GOLD")

    fm_boot.insert(0, "Dataset", "Full dataset")

    combined_fm_boot = pd.concat([fm_boot, fm_boot_highconf], ignore_index=True)

    # Write outputs
    out_sens = OUTDIR / "Dataset_S24_documentation_tier_sensitivity.csv"
    out_boot = OUTDIR / "Dataset_S25_FM_bootstrap_CI.csv"
    out_txt = OUTDIR / "Dataset_S24_S25_sensitivity_bootstrap_report.txt"

    sensitivity.to_csv(out_sens, index=False)
    combined_fm_boot.to_csv(out_boot, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Documentation-tier sensitivity analysis and FM bootstrap robustness report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input dataset: {dataset_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Bootstrap iterations: {BOOT_N}\n\n")

        f.write("Tier counts:\n")
        f.write(df["_tier_norm"].value_counts().to_string())
        f.write("\n\n")

        f.write("Sensitivity analysis:\n")
        f.write(sensitivity.to_string(index=False))
        f.write("\n\n")

        f.write("FM bootstrap CI analysis:\n")
        f.write(combined_fm_boot.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation guide:\n")
        f.write("- Negative bias = overestimation of lipophilicity by consensus logP.\n")
        f.write("- Positive bias = underestimation of lipophilicity by consensus logP.\n")
        f.write("- Bootstrap CIs are descriptive robustness intervals, not independent validation of a predictive classifier.\n")

    print("\nAnaliz tamamlandı.")
    print(f"Sensitivity CSV : {out_sens}")
    print(f"Bootstrap CSV   : {out_boot}")
    print(f"Report TXT      : {out_txt}")

    print("\n=== Sensitivity summary ===")
    print(sensitivity.to_string(index=False))

    print("\n=== FM bootstrap summary ===")
    print(combined_fm_boot.to_string(index=False))


if __name__ == "__main__":
    main()