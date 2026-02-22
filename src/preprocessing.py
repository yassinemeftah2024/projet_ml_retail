# src/preprocessing.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filename: str) -> pd.DataFrame:
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df


def quick_report(df: pd.DataFrame) -> None:
    print("\n========== DATA OVERVIEW ==========")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))

    print("\n--- Dtypes ---")
    print(df.dtypes)

    print("\n--- Missing values (top 30) ---")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss.head(30))

    dup = df.duplicated().sum()
    print("\n--- Duplicates ---")
    print("Duplicated rows:", dup)

    print("\n--- Sample ---")
    print(df.head(5))
def main():
    # ✅ put here the exact CSV filename inside data/raw/
    filename = "retail_customers_COMPLETE_CATEGORICAL.csv"

    # 1) Load
    df = load_data(filename)

    # 2) Basic analysis / overview
    quick_report(df)


if __name__ == "__main__":
    main()
