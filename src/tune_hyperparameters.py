import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.preprocess import prepare_data

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)


def get_available_cores() -> int:
    return os.cpu_count() or 4


def tune_and_optimize_coverage(
    file_path: str | None = None, output_dir: str = "figures"
) -> dict[str, Any]:
    cores = get_available_cores()
    print(
        f"\n[Hardware Acceleration] Detected {cores} CPU threads. Launching parallel search across all cores..."
    )

    X_train, X_test, y_train, y_test, _, _ = prepare_data(file_path=file_path)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gb_param_dist = {
        "learning_rate": [0.03, 0.05, 0.08, 0.10, 0.14],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [15, 25, 50, 100],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0, 3.0],
        "max_iter": [150, 200, 250],
    }

    base_gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    search_gb = RandomizedSearchCV(
        estimator=base_gb,
        param_distributions=gb_param_dist,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search_gb.fit(X_train, y_train)

    best_gb = search_gb.best_estimator_
    best_gb_cv = float(search_gb.best_score_)
    print(f" -> Best Gradient Boosting CV ROC-AUC: {best_gb_cv:.4f}")
    print(f" -> Optimal Parameters: {search_gb.best_params_}")

    joblib.dump(best_gb, os.path.join("models", "gradient_boosting.joblib"))

    test_probs = best_gb.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_probs))

    thresholds = np.linspace(0.20, 0.70, 51)
    coverage_curve = []

    for t in thresholds:
        preds = (test_probs >= t).astype(int)
        rec = float(recall_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        bal = float(balanced_accuracy_score(y_test, preds))
        acc = float(accuracy_score(y_test, preds))
        coverage_curve.append(
            {
                "Threshold": t,
                "Recall (Coverage)": rec,
                "Precision": prec,
                "Balanced Acc": bal,
                "Accuracy": acc,
            }
        )

    curve_df = pd.DataFrame(coverage_curve)

    row_75 = curve_df[curve_df["Recall (Coverage)"] >= 0.75].iloc[-1]
    row_80 = curve_df[curve_df["Recall (Coverage)"] >= 0.80].iloc[-1]

    print("\n" + "=" * 80)
    print("           OPTIMAL DECISION THRESHOLDS FOR MAXIMUM COVERAGE")
    print("=" * 80)
    print("Standard Default Threshold (0.50):")
    std_row = curve_df.iloc[(curve_df["Threshold"] - 0.50).abs().argmin()]
    print(
        f" -> Coverage (Recall): {std_row['Recall (Coverage)'] * 100:.1f}% | Precision: {std_row['Precision'] * 100:.1f}% | Balanced Acc: {std_row['Balanced Acc'] * 100:.1f}%"
    )

    print(
        f"\nTuned Underwriter Threshold ({row_75['Threshold']:.2f}) -> Target: 75% Coverage:"
    )
    print(
        f" -> Coverage (Recall): {row_75['Recall (Coverage)'] * 100:.1f}% | Precision: {row_75['Precision'] * 100:.1f}% | Balanced Acc: {row_75['Balanced Acc'] * 100:.1f}%"
    )

    print(
        f"\nAggressive Risk Shield Threshold ({row_80['Threshold']:.2f}) -> Target: 80% Coverage:"
    )
    print(
        f" -> Coverage (Recall): {row_80['Recall (Coverage)'] * 100:.1f}% | Precision: {row_80['Precision'] * 100:.1f}% | Balanced Acc: {row_80['Balanced Acc'] * 100:.1f}%"
    )
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.plot(
        curve_df["Threshold"],
        curve_df["Recall (Coverage)"] * 100,
        label="Default Coverage (Recall %)",
        color="#d9534f",
        linewidth=2.5,
    )
    plt.plot(
        curve_df["Threshold"],
        curve_df["Precision"] * 100,
        label="Precision % (True Positives / Flags)",
        color="#2b5c8f",
        linewidth=2.0,
    )
    plt.plot(
        curve_df["Threshold"],
        curve_df["Balanced Acc"] * 100,
        label="Balanced Accuracy %",
        color="#5cb85c",
        linestyle="--",
        linewidth=2.0,
    )

    plt.axvline(
        x=row_80["Threshold"],
        color="#d9534f",
        linestyle=":",
        label=f"80% Coverage Threshold ({row_80['Threshold']:.2f})",
    )
    plt.axvline(
        x=0.50,
        color="#777777",
        linestyle="--",
        alpha=0.7,
        label="Standard Default (0.50)",
    )

    plt.title(
        "Credit Risk Threshold vs. Default Coverage Curve",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Underwriting Decision Threshold (Prob. Cutoff)", fontsize=11)
    plt.ylabel("Metric Score (%)", fontsize=11)
    plt.xlim(0.20, 0.70)
    plt.ylim(20, 100)
    plt.legend(loc="center left", frameon=True)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "threshold_coverage_tradeoff.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n[Visualization] Saved threshold trade-off chart to: {chart_path}")

    return {
        "Best Model": best_gb,
        "Best CV AUC": best_gb_cv,
        "Test AUC": test_auc,
        "75% Coverage Threshold": float(row_75["Threshold"]),
        "80% Coverage Threshold": float(row_80["Threshold"]),
    }


if __name__ == "__main__":
    tune_and_optimize_coverage()
