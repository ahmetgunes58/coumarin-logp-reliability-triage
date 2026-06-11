from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Figures S2 and S3
# Predictor-level signed and absolute error distributions
# ============================================================

PROJECT_DIR = Path(r"D:\Makaleler\coumarin-logp-working-source")

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

OUT_FIG_DIR = PROJECT_DIR / "figures" / "supplementary"
OUT_SRC_DIR = PROJECT_DIR / "figures" / "source_data"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SRC_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find a column from a list of candidate names."""
    normalized = {c.lower().replace(" ", "").replace("-", "_"): c for c in df.columns}

    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("-", "_")
        if key in normalized:
            return normalized[key]

    raise KeyError(
        f"None of the candidate columns were found: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save PNG, PDF, and SVG versions."""
    png = OUT_FIG_DIR / f"{stem}.png"
    pdf = OUT_FIG_DIR / f"{stem}.pdf"
    svg = OUT_FIG_DIR / f"{stem}.svg"

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {svg}")


def make_box_jitter_plot(
    plot_df: pd.DataFrame,
    value_col: str,
    y_label: str,
    output_stem: str,
    zero_line: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Create a clean boxplot + jitter plot."""

    predictor_order = [
        "iLOGP",
        "XLOGP3",
        "WLOGP",
        "MLOGP",
        "Silicos-IT",
        "Consensus",
    ]

    values = [
        plot_df.loc[plot_df["Predictor"] == p, value_col].dropna().to_numpy()
        for p in predictor_order
    ]

    rng = np.random.default_rng(20260609)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.4, 4.9))

    ax.boxplot(
        values,
        positions=np.arange(1, len(predictor_order) + 1),
        widths=0.55,
        showfliers=False,
        patch_artist=False,
        medianprops={"linewidth": 1.2, "color": "black"},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )

    # Jittered compound-level points
    for i, arr in enumerate(values, start=1):
        x = rng.normal(loc=i, scale=0.055, size=len(arr))
        ax.scatter(
            x,
            arr,
            s=14,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )

    if zero_line:
        ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.65, zorder=1)

    ax.set_xticks(np.arange(1, len(predictor_order) + 1))
    ax.set_xticklabels(predictor_order, rotation=30, ha="right")

    ax.set_ylabel(y_label)
    ax.set_xlabel("")

    if y_limits is not None:
        ax.set_ylim(y_limits)

    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


# ------------------------------------------------------------
# Load input
# ------------------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

compound_col = find_column(df, ["Compound_ID", "compound_id", "ID"])
logp_col = find_column(df, ["logP_exp", "LogP_exp", "experimental_logP", "Exp_logP"])

predictor_columns = {
    "iLOGP": find_column(df, ["iLOGP", "ilogp"]),
    "XLOGP3": find_column(df, ["XLOGP3", "xlogp3"]),
    "WLOGP": find_column(df, ["WLOGP", "wlogp"]),
    "MLOGP": find_column(df, ["MLOGP", "mlogp"]),
    "Silicos-IT": find_column(df, ["Silicos-IT", "Silicos_IT", "SilicosIT", "silicos_it"]),
    "Consensus": find_column(df, ["Consensus", "consensus", "Consensus_logP", "consensus_logP"]),
}

# ------------------------------------------------------------
# Build long-format source data
# ΔlogP = logP_exp − logP_pred
# Negative values indicate overestimation.
# ------------------------------------------------------------
records = []

for predictor_name, pred_col in predictor_columns.items():
    temp = df[[compound_col, logp_col, pred_col]].copy()
    temp.columns = ["Compound_ID", "logP_exp", "logP_pred"]
    temp["Predictor"] = predictor_name
    temp["delta_logP"] = temp["logP_exp"] - temp["logP_pred"]
    temp["abs_delta_logP"] = temp["delta_logP"].abs()
    records.append(temp)

long_df = pd.concat(records, ignore_index=True)

# Save source-data files separately to match SI naming
signed_source = OUT_SRC_DIR / "Figure_S2_six_predictor_signed_error_source_data.csv"
absolute_source = OUT_SRC_DIR / "Figure_S3_six_predictor_absolute_error_source_data.csv"

long_df[
    ["Compound_ID", "Predictor", "logP_exp", "logP_pred", "delta_logP"]
].to_csv(signed_source, index=False, encoding="utf-8-sig")

long_df[
    ["Compound_ID", "Predictor", "logP_exp", "logP_pred", "delta_logP", "abs_delta_logP"]
].to_csv(absolute_source, index=False, encoding="utf-8-sig")

print(f"Saved source data: {signed_source}")
print(f"Saved source data: {absolute_source}")

# ------------------------------------------------------------
# Figure S2: signed prediction-error distributions
# ------------------------------------------------------------
make_box_jitter_plot(
    plot_df=long_df,
    value_col="delta_logP",
    y_label="Signed error, ΔlogP",
    output_stem="Figure_S2_six_predictor_signed_error_distributions",
    zero_line=True,
    y_limits=None,
)

# ------------------------------------------------------------
# Figure S3: absolute prediction-error distributions
# ------------------------------------------------------------
make_box_jitter_plot(
    plot_df=long_df,
    value_col="abs_delta_logP",
    y_label="Absolute error, |ΔlogP|",
    output_stem="Figure_S3_six_predictor_absolute_error_distributions",
    zero_line=False,
    y_limits=(0, None),
)

# ------------------------------------------------------------
# Quick reproducibility summary
# ------------------------------------------------------------
summary = (
    long_df.groupby("Predictor")
    .agg(
        n=("delta_logP", "count"),
        bias=("delta_logP", "mean"),
        median_error=("delta_logP", "median"),
        mae=("abs_delta_logP", "mean"),
        rmse=("delta_logP", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        severe_abs_error_percent=("abs_delta_logP", lambda x: float((x >= 2.0).mean() * 100)),
    )
    .reset_index()
)

summary_file = OUT_SRC_DIR / "Figure_S2_S3_predictor_error_distribution_summary.csv"
summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

print(f"Saved summary: {summary_file}")
print("Figures S2 and S3 generated successfully.")