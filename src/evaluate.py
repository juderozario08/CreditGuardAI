import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocess import prepare_data
from src.train_pytorch import CreditDefaultMLP

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def run_full_evaluation(
    file_path: str | None = None, output_dir: str = "figures"
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    _, X_test, _, y_test, feature_names, _ = prepare_data(file_path=file_path)

    models: dict[str, Any] = {}
    classical_names = [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Random Forest", "random_forest.joblib"),
        ("Gradient Boosting", "gradient_boosting.joblib"),
    ]
    for display_name, file_name in classical_names:
        p = os.path.join("models", file_name)
        if os.path.exists(p):
            models[display_name] = joblib.load(p)

    pytorch_path = os.path.join("models", "pytorch_credit_model.pth")
    pytorch_model = None
    if os.path.exists(pytorch_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model = CreditDefaultMLP(in_features=X_test.shape[1]).to(device)
        pytorch_model.load_state_dict(
            torch.load(pytorch_path, weights_only=True, map_location=device)
        )
        pytorch_model.eval()

    eval_results: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        eval_results[name] = {
            "y_prob": y_prob,
            "y_pred": y_pred,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Balanced Acc": float(balanced_accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
            "PR-AUC": float(average_precision_score(y_test, y_prob)),
        }

    if pytorch_model is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            t_X = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = pytorch_model(t_X)
            y_prob = torch.sigmoid(logits).cpu().numpy().flatten()
            y_pred = (y_prob >= 0.5).astype(int)

        eval_results["PyTorch Neural Net (MLP)"] = {
            "y_prob": y_prob,
            "y_pred": y_pred,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Balanced Acc": float(balanced_accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
            "PR-AUC": float(average_precision_score(y_test, y_prob)),
        }

    table_rows = []
    for model_name, metrics in eval_results.items():
        table_rows.append(
            {
                "Model Architecture": model_name,
                "Accuracy": f"{metrics['Accuracy'] * 100:.2f}%",
                "Balanced Acc": f"{metrics['Balanced Acc'] * 100:.2f}%",
                "Precision": f"{metrics['Precision'] * 100:.2f}%",
                "Recall": f"{metrics['Recall'] * 100:.2f}%",
                "F1-Score": f"{metrics['F1-Score']:.4f}",
                "ROC-AUC": f"{metrics['ROC-AUC']:.4f}",
                "PR-AUC": f"{metrics['PR-AUC']:.4f}",
            }
        )

    benchmark_df = pd.DataFrame(table_rows)
    print("\n" + "=" * 85)
    print("               CREDITGUARD AI: MASTER BENCHMARK EVALUATION")
    print("=" * 85)
    print(benchmark_df.to_string(index=False))
    print("=" * 85 + "\n")

    _, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = {
        "Logistic Regression": "#7f7f7f",
        "Random Forest": "#2ca02c",
        "Gradient Boosting": "#1f77b4",
        "PyTorch Neural Net (MLP)": "#d62728",
    }

    # 1. Combined ROC Curves
    ax_roc = axes[0, 0]
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")
    for name, res in eval_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax_roc.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {res['ROC-AUC']:.4f})",
            color=colors.get(name, "#333333"),
            linewidth=2.0,
        )
    ax_roc.set_title(
        "A. Receiver Operating Characteristic (ROC) Curves",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.grid(True, linestyle="--", alpha=0.5)

    # 2. Precision-Recall Curves
    ax_pr = axes[0, 1]
    no_skill = float(np.sum(y_test) / len(y_test))
    ax_pr.plot(
        [0, 1],
        [no_skill, no_skill],
        "k--",
        alpha=0.6,
        label=f"Baseline Rate ({no_skill:.2f})",
    )
    for name, res in eval_results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax_pr.plot(
            rec,
            prec,
            label=f"{name} (PR-AUC = {res['PR-AUC']:.4f})",
            color=colors.get(name, "#333333"),
            linewidth=2.0,
        )
    ax_pr.set_title(
        "B. Precision-Recall Curves (Class Imbalance View)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_pr.set_xlabel("Recall (Coverage of Defaults)")
    ax_pr.set_ylabel("Precision (Accuracy of Default Flag)")
    ax_pr.legend(loc="upper right", frameon=True)
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    # 3. Confusion Matrix of Top Model
    ax_cm = axes[1, 0]
    best_model_name = (
        "Gradient Boosting"
        if "Gradient Boosting" in eval_results
        else next(iter(eval_results.keys()))
    )
    cm = confusion_matrix(y_test, eval_results[best_model_name]["y_pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax_cm,
        xticklabels=["Settled (0)", "Default (1)"],
        yticklabels=["Settled (0)", "Default (1)"],
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax_cm.set_title(
        f"C. Confusion Matrix: {best_model_name}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_cm.set_xlabel("Predicted Class")
    ax_cm.set_ylabel("Actual Class")

    # 4. Top 10 Feature Importances
    ax_feat = axes[1, 1]
    if "Random Forest" in models:
        rf = models["Random Forest"]
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in sorted_idx][::-1]
        top_importances = importances[sorted_idx][::-1]

        ax_feat.barh(
            range(len(top_features)), top_importances, color="#1f77b4", align="center"
        )
        ax_feat.set_yticks(range(len(top_features)))
        ax_feat.set_yticklabels(top_features)
        ax_feat.set_title(
            "D. Top 10 Credit Risk Predictors (Feature Importance)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax_feat.set_xlabel("Relative Importance Score")
    else:
        ax_feat.text(
            0.5, 0.5, "Feature Importances Not Available", ha="center", va="center"
        )

    plt.tight_layout(pad=3.0)
    dashboard_path = os.path.join(output_dir, "model_benchmark_dashboard.png")
    plt.savefig(dashboard_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved 4-panel dashboard to: {dashboard_path}")

    return benchmark_df


if __name__ == "__main__":
    run_full_evaluation()
