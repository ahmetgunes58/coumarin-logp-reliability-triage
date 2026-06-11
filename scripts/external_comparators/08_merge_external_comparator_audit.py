# -*- coding: utf-8 -*-
"""
08_merge_external_comparator_audit.py

Purpose
-------
Merge all external platform-independent logP comparator outputs and generate
the final external-comparator audit tables.

Inputs
------
data/processed/Dataset_S42_external_predictor_input.csv
data/processed/Dataset_S42a_RDKit_MolLogP_values.csv
data/processed/Dataset_S42b_DataWarrior_cLogP_values.csv
data/processed/Dataset_S42c_ALOGPS_2_1_values.csv
data/processed/Dataset_S42d_OPERA_logP_values.csv

Outputs
-------
data/processed/Dataset_S42_external_predictor_values.csv
data/processed/Dataset_S43_external_predictor_performance.csv
data/processed/Dataset_S44_external_high_risk_regime_summary.csv
data/processed/Dataset_S45_external_predictor_audit_report.txt

Important
---------
Dataset_S1 remains the frozen benchmark dataset. This script creates an
external comparator audit only. It does not train, recalibrate, or replace
any SwissADME-associated predictor.
"""

from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]

BASE_INPUT = PROJECT_DIR / "data" / "processed" / "Dataset_S42_external_predictor_input.csv"

RDKIT_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S42a_RDKit_MolLogP_values.csv"
DW_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S42b_DataWarrior_cLogP_values.csv"
ALOGPS_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S42c_ALOGPS_2_1_values.csv"
OPERA_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S42d_OPERA_logP_values.csv"

OUT_VALUES = PROJECT_DIR / "data" / "processed" / "Dataset_S42_external_predictor_values.csv"
OUT_PERFORMANCE = PROJECT_DIR / "data" / "processed" / "Dataset_S43_external_predictor_performance.csv"
OUT_HIGH_RISK = PROJECT_DIR / "data" / "processed" / "Dataset_S44_external_high_risk_regime_summary.csv"
OUT_REPORT = PROJECT_DIR / "data" / "processed" / "Dataset_S45_external_predictor_audit_report.txt"


PREDICTORS = {
    "SwissADME_Consensus": {
        "column": "Consensus",
        "group": "Primary SwissADME-associated reference",
    },
    "RDKit_MolLogP": {
        "column": "RDKit_MolLogP",
        "group": "External comparator",
    },
    "DataWarrior_cLogP": {
        "column": "DataWarrior_cLogP",
        "group": "External comparator",
    },
    "ALOGPS_2_1_logP": {
        "column": "ALOGPS_2_1_logP",
        "group": "External comparator",
    },
    "OPERA_logP": {
        "column": "OPERA_logP",
        "group": "External comparator",
    },
}


def calculate_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def calculate_metrics(df: pd.DataFrame, predictor_col: str, mask=None) -> dict:
    if mask is None:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask].copy()
    valid = sub["logP_exp"].notna() & sub[predictor_col].notna()

    if valid.sum() == 0:
        return {
            "n": 0,
            "Bias": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Overestimated_percent": np.nan,
            "Underestimated_percent": np.nan,
            "Severe_abs_error_percent": np.nan,
            "Severe_overestimation_percent": np.nan,
            "Severe_underestimation_percent": np.nan,
        }

    y_true = sub.loc[valid, "logP_exp"].astype(float)
    y_pred = sub.loc[valid, predictor_col].astype(float)
    delta = y_true - y_pred

    return {
        "n": int(valid.sum()),
        "Bias": float(delta.mean()),
        "MAE": float(delta.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(delta ** 2))),
        "R2": calculate_r2(y_true, y_pred),
        "Overestimated_percent": float((delta < 0).mean() * 100),
        "Underestimated_percent": float((delta > 0).mean() * 100),
        "Severe_abs_error_percent": float((delta.abs() >= 2.0).mean() * 100),
        "Severe_overestimation_percent": float((delta <= -2.0).mean() * 100),
        "Severe_underestimation_percent": float((delta >= 2.0).mean() * 100),
    }


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def main() -> None:
    for path in [BASE_INPUT, RDKIT_FILE, DW_FILE, ALOGPS_FILE, OPERA_FILE]:
        require_file(path)

    base = pd.read_csv(BASE_INPUT, encoding="utf-8-sig")
    rdkit = pd.read_csv(RDKIT_FILE, encoding="utf-8-sig")
    dw = pd.read_csv(DW_FILE, encoding="utf-8-sig")
    alogps = pd.read_csv(ALOGPS_FILE, encoding="utf-8-sig")
    opera = pd.read_csv(OPERA_FILE, encoding="utf-8-sig")

    required_base = [
        "Compound_ID", "SMILES", "logP_exp", "N_count", "N_group",
        "FM", "Data_Tier", "Consensus", "delta_Consensus"
    ]
    missing_base = [c for c in required_base if c not in base.columns]
    if missing_base:
        raise ValueError(f"Missing columns in base input: {missing_base}")

    if base["Compound_ID"].duplicated().any():
        raise ValueError("Duplicate Compound_ID values detected in base input.")

    # Start from frozen external input.
    df = base[required_base + [c for c in ["logP_range", "logP_15", "Coumarin_Type"] if c in base.columns]].copy()

    # Merge RDKit.
    df = df.merge(
        rdkit[["Compound_ID", "RDKit_MolLogP", "delta_RDKit_MolLogP", "abs_delta_RDKit_MolLogP", "RDKit_status"]],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )

    # Merge DataWarrior.
    df = df.merge(
        dw[["Compound_ID", "DataWarrior_cLogP", "delta_DataWarrior_cLogP", "abs_delta_DataWarrior_cLogP"]],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )

    # Merge ALOGPS.
    df = df.merge(
        alogps[
            [
                "Compound_ID",
                "ALOGPS_2_1_logP",
                "ALOGPS_logS",
                "delta_ALOGPS_2_1_logP",
                "abs_delta_ALOGPS_2_1_logP",
            ]
        ],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )

    # Merge OPERA.
    df = df.merge(
        opera[
            [
                "Compound_ID",
                "OPERA_logP",
                "OPERA_logP_predRange",
                "OPERA_AD_LogP",
                "OPERA_AD_index_LogP",
                "OPERA_Conf_index_LogP",
                "delta_OPERA_logP",
                "abs_delta_OPERA_logP",
            ]
        ],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
    )

    # Numeric coercion and final delta recalculation for consistency.
    df["logP_exp"] = pd.to_numeric(df["logP_exp"], errors="raise")
    df["N_count"] = pd.to_numeric(df["N_count"], errors="raise")
    df["Consensus"] = pd.to_numeric(df["Consensus"], errors="raise")

    for predictor_name, meta in PREDICTORS.items():
        col = meta["column"]
        df[col] = pd.to_numeric(df[col], errors="raise")
        delta_col = f"delta_{predictor_name}"
        abs_delta_col = f"abs_delta_{predictor_name}"
        df[delta_col] = df["logP_exp"] - df[col]
        df[abs_delta_col] = df[delta_col].abs()

    # Regime flags used for compact manuscript and SI summaries.
    df["flag_polar_logP_le_1"] = df["logP_exp"] <= 1.0
    df["flag_N_1_3_logP_lt_1_5"] = df["N_count"].between(1, 3) & (df["logP_exp"] < 1.5)
    df["flag_severe_consensus_abs_error"] = df["delta_SwissADME_Consensus"].abs() >= 2.0
    df["flag_FM4_exploratory_boundary"] = df["FM"].astype(str).eq("FM4")

    OUT_VALUES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_VALUES, index=False, encoding="utf-8")

    # -----------------------------------------------------------------
    # Dataset_S43: overall predictor performance
    # -----------------------------------------------------------------
    perf_rows = []

    for predictor_name, meta in PREDICTORS.items():
        row = {
            "Predictor": predictor_name,
            "Predictor_group": meta["group"],
            "Prediction_column": meta["column"],
        }
        row.update(calculate_metrics(df, meta["column"]))
        perf_rows.append(row)

    performance = pd.DataFrame(perf_rows)
    performance.to_csv(OUT_PERFORMANCE, index=False, encoding="utf-8")

    # -----------------------------------------------------------------
    # Dataset_S44: high-risk regime summaries
    # -----------------------------------------------------------------
    subset_masks = {
        "Full dataset": pd.Series(True, index=df.index),
        "Polar logP <= 1.0": df["flag_polar_logP_le_1"],
        "N = 1-3 and logP < 1.5": df["flag_N_1_3_logP_lt_1_5"],
        "FM1 severe overestimation regime": df["FM"].astype(str).eq("FM1"),
        "FM2 intermediate / high-disagreement regime": df["FM"].astype(str).eq("FM2"),
        "FM3 mixed high-N regime": df["FM"].astype(str).eq("FM3"),
        "FM4 exploratory boundary observation": df["FM"].astype(str).eq("FM4"),
    }

    # OPERA-specific AD subsets are included for OPERA only in the same table.
    high_rows = []

    for predictor_name, meta in PREDICTORS.items():
        for subset_name, mask in subset_masks.items():
            row = {
                "Predictor": predictor_name,
                "Predictor_group": meta["group"],
                "Subset": subset_name,
            }
            row.update(calculate_metrics(df, meta["column"], mask))
            high_rows.append(row)

    for subset_name, mask in {
        "OPERA AD_LogP = 1": df["OPERA_AD_LogP"].eq(1),
        "OPERA AD_LogP = 0": df["OPERA_AD_LogP"].eq(0),
    }.items():
        row = {
            "Predictor": "OPERA_logP",
            "Predictor_group": "External comparator",
            "Subset": subset_name,
        }
        row.update(calculate_metrics(df, "OPERA_logP", mask))
        high_rows.append(row)

    high_risk = pd.DataFrame(high_rows)
    high_risk.to_csv(OUT_HIGH_RISK, index=False, encoding="utf-8")

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    report = []
    report.append("Dataset_S45 external platform-independent logP comparator audit report")
    report.append("=" * 80)
    report.append("")
    report.append(f"Project directory: {PROJECT_DIR}")
    report.append(f"Output values: {OUT_VALUES}")
    report.append(f"Output performance table: {OUT_PERFORMANCE}")
    report.append(f"Output high-risk summary: {OUT_HIGH_RISK}")
    report.append(f"Output report: {OUT_REPORT}")
    report.append("")
    report.append("1. Input integrity")
    report.append("-" * 80)
    report.append(f"Compounds in merged audit table: {len(df)}")
    report.append(f"Unique Compound_ID values: {df['Compound_ID'].nunique()}")
    report.append("")
    report.append("Missing values by predictor:")
    for predictor_name, meta in PREDICTORS.items():
        col = meta["column"]
        report.append(f"  {predictor_name}: {int(df[col].isna().sum())}")
    report.append("")
    report.append("2. Overall predictor performance")
    report.append("-" * 80)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        report.append(performance.to_string(index=False))
    report.append("")
    report.append("3. Key high-risk regime summary")
    report.append("-" * 80)

    key_subsets = high_risk[
        high_risk["Subset"].isin(
            [
                "Full dataset",
                "Polar logP <= 1.0",
                "N = 1-3 and logP < 1.5",
                "FM1 severe overestimation regime",
                "FM4 exploratory boundary observation",
            ]
        )
    ].copy()

    with pd.option_context("display.max_columns", None, "display.width", 220):
        report.append(key_subsets.to_string(index=False))

    report.append("")
    report.append("4. Main interpretation for manuscript use")
    report.append("-" * 80)
    report.append(
        "The external comparator audit shows that the error is not equally severe "
        "across all logP engines. OPERA shows near-zero full-dataset signed bias, "
        "whereas RDKit MolLogP, DataWarrior cLogP, and ALOGPS 2.1 retain negative "
        "full-dataset bias. However, the polar logP <= 1.0 and N = 1-3 / logP < 1.5 "
        "regimes show directional overestimation across multiple non-SwissADME "
        "comparators. This supports the interpretation that the high-risk regime is "
        "not solely a SwissADME-specific artefact, while preserving the conclusion "
        "that logP-prediction reliability is tool- and regime-dependent."
    )
    report.append("")
    report.append("5. Scope note")
    report.append("-" * 80)
    report.append(
        "This audit is not a new model-training exercise, not a full multi-tool logP "
        "benchmark, and not an external prospective validation of the FM0-FM4 taxonomy. "
        "It is a compact platform-independent comparator audit designed to test whether "
        "the SwissADME-associated directional failure pattern is confined to the "
        "SwissADME suite."
    )

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("SUCCESS: External comparator audit merge completed.")


if __name__ == "__main__":
    main()