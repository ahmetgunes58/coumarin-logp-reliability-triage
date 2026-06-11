from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DATA = ROOT / "data" / "processed"
DATA.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    DATA / "Dataset_S1_benchmark_dataset.csv",
    ROOT / "Dataset_S1_benchmark_dataset.csv",
    ROOT / "data" / "Dataset_S1_benchmark_dataset.csv",
]

PREDICTORS = ["iLOGP", "XLOGP3", "WLOGP", "MLOGP", "Silicos_IT", "Consensus"]

OUT_PERF = DATA / "Dataset_S26_outlier_robustness_predictor_performance.csv"
OUT_REMOVED = DATA / "Dataset_S27_outlier_removed_compounds.csv"
OUT_REPORT = DATA / "Dataset_S26_S27_outlier_robustness_report.txt"


def find_input_file():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset_S1_benchmark_dataset.csv bulunamadı. "
        "Dosyayı data/processed içine koy."
    )


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def performance_metrics(df, predictor, dataset_label):
    exp = pd.to_numeric(df["logP_exp"], errors="coerce")
    pred = pd.to_numeric(df[predictor], errors="coerce")
    delta = exp - pred

    mask = exp.notna() & pred.notna()
    exp = exp[mask]
    pred = pred[mask]
    delta = delta[mask]

    return {
        "Dataset": dataset_label,
        "Predictor": predictor,
        "n": int(len(delta)),
        "Bias_mean_Exp_minus_Pred": float(delta.mean()),
        "Bias_median_Exp_minus_Pred": float(delta.median()),
        "MAE": float(np.mean(np.abs(delta))),
        "RMSE": float(np.sqrt(np.mean(delta ** 2))),
        "R2": float(r2_score(exp, pred)),
        "Overestimated_n": int((delta < 0).sum()),
        "Overestimated_percent": float(100 * (delta < 0).mean()),
        "Severe_error_abs_ge_2_n": int((np.abs(delta) >= 2.0).sum()),
        "Severe_error_abs_ge_2_percent": float(100 * (np.abs(delta) >= 2.0).mean()),
        "Max_abs_error": float(np.max(np.abs(delta))),
    }


def make_removed_table(df):
    exp = pd.to_numeric(df["logP_exp"], errors="coerce")
    pred = pd.to_numeric(df["Consensus"], errors="coerce")
    delta = exp - pred

    tmp = df[["Compound_ID", "logP_exp", "Consensus", "N_count", "FM"]].copy()
    tmp["delta_Consensus"] = delta
    tmp["abs_delta_Consensus"] = np.abs(delta)
    tmp = tmp.sort_values("abs_delta_Consensus", ascending=False).reset_index(drop=True)
    tmp["abs_error_rank"] = np.arange(1, len(tmp) + 1)

    return tmp


def subset_definitions(df, removed_table):
    subsets = []

    subsets.append(("Full dataset", df.copy(), []))

    no_058 = df[df["Compound_ID"] != "CMR_GOLD_058"].copy()
    subsets.append(("Excluding CMR_GOLD_058", no_058, ["CMR_GOLD_058"]))

    for k in [1, 3, 5]:
        ids = removed_table.head(k)["Compound_ID"].tolist()
        sub = df[~df["Compound_ID"].isin(ids)].copy()
        subsets.append((f"Excluding top-{k} absolute consensus-error compound(s)", sub, ids))

    return subsets


def main():
    input_file = find_input_file()
    df = pd.read_csv(input_file)

    required = ["Compound_ID", "logP_exp", "Consensus", "N_count", "FM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Eksik gerekli kolonlar: {missing}. Mevcut kolonlar: {list(df.columns)}")

    missing_pred = [p for p in PREDICTORS if p not in df.columns]
    if missing_pred:
        raise KeyError(f"Eksik predictor kolonları: {missing_pred}")

    removed_table = make_removed_table(df)
    subsets = subset_definitions(df, removed_table)

    perf_rows = []
    removal_rows = []

    for label, sub, removed_ids in subsets:
        for predictor in PREDICTORS:
            perf_rows.append(performance_metrics(sub, predictor, label))

        for rid in removed_ids:
            row = removed_table[removed_table["Compound_ID"] == rid].iloc[0].to_dict()
            row["Removal_scenario"] = label
            removal_rows.append(row)

    perf = pd.DataFrame(perf_rows)
    removed = pd.DataFrame(removal_rows)

    perf.to_csv(OUT_PERF, index=False, encoding="utf-8-sig")
    removed.to_csv(OUT_REMOVED, index=False, encoding="utf-8-sig")

    # Compact consensus-only table for screen/report
    consensus = perf[perf["Predictor"] == "Consensus"].copy()
    cols = [
        "Dataset", "n", "Bias_mean_Exp_minus_Pred", "Bias_median_Exp_minus_Pred",
        "MAE", "RMSE", "R2", "Overestimated_percent",
        "Severe_error_abs_ge_2_percent", "Max_abs_error"
    ]

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Outlier-robustness analysis for SwissADME-associated logP prediction\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Full dataset n: {len(df)}\n")
        f.write("Error convention: delta = experimental logP - predicted logP\n")
        f.write("Negative bias indicates overestimation of lipophilicity.\n\n")

        f.write("Top absolute consensus-error compounds:\n")
        f.write(
            removed_table.head(10)[
                ["abs_error_rank", "Compound_ID", "logP_exp", "Consensus",
                 "delta_Consensus", "abs_delta_Consensus", "N_count", "FM"]
            ].to_string(index=False)
        )
        f.write("\n\n")

        f.write("Consensus outlier-robustness summary:\n")
        f.write(consensus[cols].to_string(index=False))
        f.write("\n\n")

        f.write("All-predictor output written to:\n")
        f.write(str(OUT_PERF) + "\n")
        f.write("Removed-compound output written to:\n")
        f.write(str(OUT_REMOVED) + "\n")

    print("\nOutlier robustness analysis tamamlandı.")
    print(f"Input      : {input_file}")
    print(f"Performance: {OUT_PERF}")
    print(f"Removed IDs: {OUT_REMOVED}")
    print(f"Report     : {OUT_REPORT}")

    print("\n=== Top 10 absolute consensus-error compounds ===")
    print(
        removed_table.head(10)[
            ["abs_error_rank", "Compound_ID", "logP_exp", "Consensus",
             "delta_Consensus", "abs_delta_Consensus", "N_count", "FM"]
        ].to_string(index=False)
    )

    print("\n=== Consensus outlier-robustness summary ===")
    print(consensus[cols].to_string(index=False))


if __name__ == "__main__":
    main()