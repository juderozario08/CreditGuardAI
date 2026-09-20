import os
from pathlib import Path

import pandas as pd


def load_raw_data(file_path: str | None = None) -> pd.DataFrame:
    if file_path is None:
        candidates = [
            "default of credit card clients.csv",
            os.path.join("data", "default of credit card clients.csv"),
            "default of credit card clients.xls",
            os.path.join("data", "default of credit card clients.xls"),
            "default of credit card clients.xlsx",
            os.path.join("data", "default of credit card clients.xlsx"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                file_path = candidate
                break
        if file_path is None:
            raise FileNotFoundError(
                "Could not find default of credit card clients dataset (.csv or .xls)."
            )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Specified data file not found: {file_path}")

    ext = Path(file_path).suffix.lower()
    print(f"[DataLoader] Ingesting dataset from: {file_path} (Format: {ext})")

    if ext == ".csv":
        df = pd.read_csv(file_path, header=1)
        if "default payment next month" not in df.columns:
            df = pd.read_csv(file_path, header=0)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, header=1)
        if "default payment next month" not in df.columns:
            df = pd.read_excel(file_path, header=0)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Must be .csv, .xls, or .xlsx."
        )

    df.columns = [str(col).strip() for col in df.columns]

    target_candidates = [
        "default payment next month",
        "Y",
        "DEFAULT_PAYMENT_NEXT_MONTH",
        "default",
    ]
    renamed = False
    for tc in target_candidates:
        if tc in df.columns:
            df.rename(columns={tc: "DEFAULT"}, inplace=True)
            renamed = True
            break

    if not renamed:
        raise KeyError(
            f"Target column not found in dataset columns: {list(df.columns)}"
        )

    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        df.rename(columns={"PAY_1": "PAY_0"}, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].apply(
            lambda x: 4 if x in [0, 5, 6] else int(x)
        )

    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: 3 if x == 0 else int(x))

    df["DEFAULT"] = df["DEFAULT"].astype(int)

    default_pct = (df["DEFAULT"].sum() / len(df)) * 100
    print(
        f"[DataLoader] Loaded {df.shape[0]} rows, {df.shape[1]} columns (Default: {default_pct:.1f}%, Settled: {100 - default_pct:.1f}%)"
    )

    return df


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    loaded_df = load_raw_data(path)
    print(loaded_df.head(3))
