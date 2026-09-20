import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data
from src.preprocess import engineer_features

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def generate_eda_figures(
    file_path: str | None = None, output_dir: str = "figures"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = load_raw_data(file_path)
    df_feat = engineer_features(df)

    # 1. Class distribution
    plt.figure(figsize=(7, 5))
    counts = df["DEFAULT"].value_counts()
    percentages = df["DEFAULT"].value_counts(normalize=True) * 100

    palette = ["#2b5c8f", "#d9534f"]
    ax = sns.barplot(
        x=counts.index, y=counts.values, palette=palette, hue=counts.index, legend=False
    )
    plt.title(
        "Credit Card Default Class Distribution (UCI Dataset #350)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Default Next Month (0 = Settled, 1 = Defaulted)", fontsize=11)
    plt.ylabel("Number of Accounts", fontsize=11)
    plt.xticks([0, 1], ["Settled / Non-Default\n(77.9%)", "Defaulted\n(22.1%)"])

    for i, count in enumerate(counts.values):
        ax.annotate(
            f"{count:,} ({percentages.iloc[i]:.1f}%)",
            xy=(i, count),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "eda_class_distribution.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()

    # 2. Delinquency (PAY_0) vs. Default Rate
    plt.figure(figsize=(9, 5))
    delinquency_risk = (
        df.groupby("PAY_0")["DEFAULT"].agg(["count", "mean"]).reset_index()
    )
    delinquency_risk = delinquency_risk[delinquency_risk["count"] >= 20]
    delinquency_risk["default_rate_pct"] = delinquency_risk["mean"] * 100

    ax2 = sns.barplot(
        data=delinquency_risk,
        x="PAY_0",
        y="default_rate_pct",
        palette="Reds_r",
        hue="PAY_0",
        legend=False,
    )
    plt.title(
        "Default Probability by Recent Repayment Status (PAY_0)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel(
        "Repayment Status (-2=No Bill, -1=Paid Full, 0=Revolving, 1-8=Months Late)",
        fontsize=11,
    )
    plt.ylabel("Observed Default Rate (%)", fontsize=11)

    for p in ax2.patches:
        val = p.get_height()
        if val > 0:
            ax2.annotate(
                f"{val:.1f}%",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="center",
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=9,
            )

    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "eda_delinquency_vs_default.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()

    # 3. Credit Utilization Distribution
    plt.figure(figsize=(8, 5))
    viz_df = df_feat[
        (df_feat["AVG_UTILIZATION"] >= -0.2) & (df_feat["AVG_UTILIZATION"] <= 1.5)
    ].copy()
    viz_df["Outcome"] = viz_df["DEFAULT"].map({0: "Settled", 1: "Defaulted"})

    sns.boxplot(
        data=viz_df,
        x="Outcome",
        y="AVG_UTILIZATION",
        palette=["#2b5c8f", "#d9534f"],
        hue="Outcome",
        legend=False,
    )
    plt.title(
        "Average 6-Month Credit Utilization vs. Default Outcome",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Account Status", fontsize=11)
    plt.ylabel("Credit Utilization (Bill / Limit)", fontsize=11)
    plt.tight_layout()
    plot_path3 = os.path.join(output_dir, "eda_utilization_distribution.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    core_cols = [
        "DEFAULT",
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "MAX_DELINQUENCY",
        "DELINQUENCY_TREND",
        "LATE_MONTHS_COUNT",
        "AVG_UTILIZATION",
        "AVG_PAY_RATIO",
        "TOTAL_PAY_VS_TOTAL_BILL",
    ]
    corr_matrix = df_feat[core_cols].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    plt.title("Risk Factors Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_path4 = os.path.join(output_dir, "eda_correlation_matrix.png")
    plt.savefig(plot_path4, dpi=300)
    plt.close()

    print(f"[EDA] Generated 4 analysis figures in '{output_dir}/'.")


if __name__ == "__main__":
    generate_eda_figures()
