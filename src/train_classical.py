import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.preprocess import prepare_data


def train_and_benchmark_classical_models(
    file_path: str | None = None, save_models_dir: str = "models"
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    os.makedirs(save_models_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, _, _ = prepare_data(file_path=file_path)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 80)
    print("      BENCHMARKING CLASSICAL MACHINE LEARNING CLASSIFIERS")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n[Training] Fitting {name} on {X_train.shape[0]} samples...")

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mean_cv_auc = float(np.mean(cv_scores))
        std_cv_auc = float(np.std(cv_scores))

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))

        results[name] = {
            "Accuracy": acc,
            "Balanced Accuracy": bal_acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "CV ROC-AUC Mean": mean_cv_auc,
            "CV ROC-AUC Std": std_cv_auc,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

        safe_filename = name.lower().replace(" ", "_") + ".joblib"
        model_save_path = os.path.join(save_models_dir, safe_filename)
        joblib.dump(model, model_save_path)
        print(
            f" -> Test ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f} | 5-Fold CV AUC: {mean_cv_auc:.4f} (±{std_cv_auc:.4f})"
        )
        print(f" -> Model artifact saved to: {model_save_path}")

    summary_rows = []
    for name, m in results.items():
        summary_rows.append(
            {
                "Classifier": name,
                "Accuracy": f"{m['Accuracy'] * 100:.2f}%",
                "Balanced Acc": f"{m['Balanced Accuracy'] * 100:.2f}%",
                "Precision": f"{m['Precision'] * 100:.2f}%",
                "Recall": f"{m['Recall'] * 100:.2f}%",
                "F1-Score": f"{m['F1-Score']:.4f}",
                "ROC-AUC": f"{m['ROC-AUC']:.4f}",
                "PR-AUC": f"{m['PR-AUC']:.4f}",
                "5-Fold CV AUC": f"{m['CV ROC-AUC Mean']:.4f} (±{m['CV ROC-AUC Std']:.3f})",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("                    FINAL CLASSICAL MODEL COMPARISON")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    return models, results, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_and_benchmark_classical_models()
