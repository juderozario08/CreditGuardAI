import argparse
import os

import joblib
import pandas as pd

from src.eda_visualizations import generate_eda_figures
from src.evaluate import run_full_evaluation
from src.train_classical import train_and_benchmark_classical_models
from src.train_pytorch import train_pytorch_model
from src.tune_hyperparameters import tune_and_optimize_coverage


def predict_sample(
    sample_dict: dict[str, float], threshold: float = 0.50
) -> dict[str, str]:
    scaler_path = os.path.join("models", "scaler.joblib")
    feature_names_path = os.path.join("models", "feature_names.joblib")
    gb_model_path = os.path.join("models", "gradient_boosting.joblib")

    if not os.path.exists(scaler_path) or not os.path.exists(gb_model_path):
        raise FileNotFoundError(
            "Trained models not found. Run --all, --train-classical, or --tune first."
        )

    scaler = joblib.load(scaler_path)
    feature_names: list[str] = joblib.load(feature_names_path)
    model = joblib.load(gb_model_path)

    full_sample = {feat: 0.0 for feat in feature_names}
    full_sample.update(sample_dict)

    input_df = pd.DataFrame([full_sample])[feature_names]
    scaled_input = scaler.transform(input_df)

    default_prob = float(model.predict_proba(scaled_input)[0, 1])
    is_default = default_prob >= threshold

    risk_tier = "LOW RISK (Approved)"
    if default_prob >= 0.60:
        risk_tier = "HIGH RISK (Declined)"
    elif is_default:
        risk_tier = f"ELEVATED RISK (Declined at {threshold:.2f} threshold)"
    elif default_prob >= (threshold - 0.10):
        risk_tier = "MODERATE RISK (Manual Review)"

    action = (
        "Reject Application"
        if is_default
        else (
            "Review Pay Stubs & Assets"
            if default_prob >= (threshold - 0.10)
            else "Approve Credit Line"
        )
    )

    return {
        "Default Probability": f"{default_prob * 100:.1f}%",
        "Operating Threshold": f"{threshold:.2f}",
        "Prediction": "DEFAULT" if is_default else "SETTLED",
        "Risk Assessment": risk_tier,
        "Recommended Action": action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CreditGuard AI: End-to-End Credit Default Prediction & Benchmarking Suite"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Path to input data file (.csv or .xls)"
    )
    parser.add_argument(
        "--eda", action="store_true", help="Generate Exploratory Data Analysis figures"
    )
    parser.add_argument(
        "--train-classical",
        action="store_true",
        help="Train and benchmark Scikit-Learn classical models",
    )
    parser.add_argument(
        "--train-pytorch", action="store_true", help="Train PyTorch Deep Learning MLP"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run master evaluation and generate 4-panel dashboard",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Execute parallel hyperparameter tuning across all CPU threads",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Custom decision threshold for coverage (e.g. 0.37 for 80% coverage)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (NVIDIA CUDA or AMD DirectML)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Execute complete pipeline end-to-end"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run a real-time risk assessment test case"
    )

    args = parser.parse_args()

    if not (
        args.eda
        or args.train_classical
        or args.train_pytorch
        or args.evaluate
        or args.tune
        or args.demo
    ):
        args.all = True

    print("=" * 80)
    print("      CreditGuard AI: Credit Default Prediction Pipeline")
    print("=" * 80)

    if args.all or args.eda:
        print("\n>>> STEP 1: Exploratory Data Analysis (EDA)")
        generate_eda_figures(file_path=args.file)

    if args.all or args.train_classical:
        print("\n>>> STEP 2: Classical Machine Learning Benchmark")
        train_and_benchmark_classical_models(file_path=args.file)

    if args.tune:
        print(
            "\n>>> STEP 2B: Multi-Threaded Hyperparameter Optimization (All CPU Cores)"
        )
        tune_and_optimize_coverage(file_path=args.file)

    if args.all or args.train_pytorch:
        print("\n>>> STEP 3: PyTorch Deep Learning Neural Network Training")
        train_pytorch_model(file_path=args.file, use_gpu=args.gpu)

    if args.all or args.evaluate:
        print("\n>>> STEP 4: Master Evaluation & Dashboard Generation")
        run_full_evaluation(file_path=args.file, use_gpu=args.gpu)

    if args.all or args.demo:
        print("\n>>> STEP 5: Real-Time Underwriter Inference Demonstration")
        sample_applicant = {
            "LIMIT_BAL": 20000.0,
            "AGE": 25.0,
            "PAY_0": 2.0,
            "PAY_2": 2.0,
            "UTILIZATION_1": 0.95,
            "MAX_DELINQUENCY": 2.0,
            "LATE_MONTHS_COUNT": 4.0,
        }
        assessment = predict_sample(sample_applicant, threshold=args.threshold)
        print("Applicant Profile:", sample_applicant)
        for k, v in assessment.items():
            print(f" -> {k}: {v}")

    print("\n" + "=" * 80)
    print("      CreditGuard AI: Execution Successfully Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
