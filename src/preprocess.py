import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_raw_data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    limit = df["LIMIT_BAL"].clip(lower=1.0)

    for i in range(1, 7):
        df[f"UTILIZATION_{i}"] = (df[f"BILL_AMT{i}"] / limit).clip(-1.0, 5.0)

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    delay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
    df["AVG_PAY_AMT"] = df[pay_cols].mean(axis=1)
    df["AVG_UTILIZATION"] = df[[f"UTILIZATION_{i}" for i in range(1, 7)]].mean(axis=1)

    for i in range(1, 6):
        prior_bill = df[f"BILL_AMT{i + 1}"].clip(lower=1.0)
        df[f"PAY_RATIO_{i}"] = (df[f"PAY_AMT{i}"] / prior_bill).clip(0.0, 10.0)

    df["AVG_PAY_RATIO"] = df[[f"PAY_RATIO_{i}" for i in range(1, 6)]].mean(axis=1)
    df["MAX_DELINQUENCY"] = df[delay_cols].max(axis=1)
    df["MIN_DELINQUENCY"] = df[delay_cols].min(axis=1)
    df["DELINQUENCY_TREND"] = df["PAY_0"] - df["PAY_6"]
    df["LATE_MONTHS_COUNT"] = (df[delay_cols] > 0).sum(axis=1)
    df["NET_BALANCE_CHANGE"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["TOTAL_PAY_VS_TOTAL_BILL"] = (
        df[pay_cols].sum(axis=1) / (df[bill_cols].sum(axis=1).clip(lower=1.0))
    ).clip(0.0, 10.0)

    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    for cat in cat_cols:
        if cat in df.columns:
            df[cat] = df[cat].astype("category")

    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def prepare_data(
    file_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    raw_df = load_raw_data(file_path)
    featured_df = engineer_features(raw_df)

    X = featured_df.drop(columns=["DEFAULT"])
    y = featured_df["DEFAULT"].to_numpy(dtype=np.int64)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_artifacts:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
        joblib.dump(feature_names, os.path.join("models", "feature_names.joblib"))

    print(
        f"[Preprocessing] Split {len(X_train)} train / {len(X_test)} test samples with {len(feature_names)} engineered features."
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    prepare_data()
