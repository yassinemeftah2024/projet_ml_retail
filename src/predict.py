from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

from utils import (
    parse_registration_date,
    ip_feature_engineering,
    handle_outliers_iqr,
    add_feature_engineering,
)

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
MODELS_DIR      = PROJECT_ROOT / "models"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Same columns as preprocessing.py
OUTLIER_COLS = ["SupportTicketsCount", "SatisfactionScore"]


def load_outlier_bounds() -> dict | None:
    bounds_path = MODELS_DIR / "outlier_bounds.json"
    if bounds_path.exists():
        return json.loads(bounds_path.read_text(encoding="utf-8"))
    return None


def load_dropped_features() -> list[str]:
    dropped_path = PROJECT_ROOT / "reports" / "dropped_features.json"
    if dropped_path.exists():
        obj = json.loads(dropped_path.read_text(encoding="utf-8"))
        return obj.get("dropped_all", [])
    return []


def prepare_features(df: pd.DataFrame, date_format: str | None = None) -> pd.DataFrame:
    """
    Apply the same feature preparation logic as preprocessing.py
    (excluding train/test split, SMOTE, and preprocessor fitting).
    """
    df = df.copy()

    # Remove unused / leakage-prone columns
    if "NewsletterSubscribed" in df.columns:
        df = df.drop(columns=["NewsletterSubscribed"])

    if "ChurnRiskCategory" in df.columns:
        df = df.drop(columns=["ChurnRiskCategory"])

    for c in list(df.columns):
        if "churnrisk" in c.lower():
            df = df.drop(columns=[c])

    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Parsing + feature engineering
    df = parse_registration_date(df, col="RegistrationDate", fmt=date_format)
    df = ip_feature_engineering(df, col="LastLoginIP")

    # Remove leakage-related columns exactly as in preprocessing.py
    leak_cols = ["Recency", "TenureRatio", "MonetaryPerDay", "CustomerType", "RFMSegment"]
    for c in leak_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Derived features
    df = add_feature_engineering(df)

    return df


def main(
    input_csv: str,
    output_csv: str,
    target: str = "Churn",
    date_format: str | None = None,
) -> None:
    input_path = Path(input_csv)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_csv

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load trained artifacts
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "best_model.joblib")

    # Load input data
    df_raw = pd.read_csv(input_path)
    df_original = df_raw.copy()

    # Keep ground truth if available
    y_true = None
    if target in df_raw.columns:
        y_true = df_raw[target].copy()
        df_raw = df_raw.drop(columns=[target])

    # Apply same preparation as preprocessing.py
    X = prepare_features(df_raw, date_format=date_format)

    # Apply same IQR clipping as train
    outlier_bounds = load_outlier_bounds()
    if outlier_bounds is not None:
        X, _ = handle_outliers_iqr(X, OUTLIER_COLS, bounds=outlier_bounds)
    else:
        print("⚠️  outlier_bounds.json not found in models/. IQR clipping skipped.")

    # Drop same train-time removed features
    dropped_features = load_dropped_features()
    if dropped_features:
        X = X.drop(columns=dropped_features, errors="ignore")

    # Align columns exactly with preprocessor expectations
    expected_cols = list(getattr(preprocessor, "feature_names_in_", []))
    if expected_cols:
        for c in expected_cols:
            if c not in X.columns:
                X[c] = pd.NA
        X = X[expected_cols]

    # Transform + predict
    Xt = preprocessor.transform(X)
    y_pred = model.predict(Xt)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xt)[:, 1]

    # Build output
    out = df_original.copy()

    if target in out.columns:
        out = out.drop(columns=[target])

    out["pred_churn"] = pd.Series(y_pred, index=out.index).astype(int)

    if y_proba is not None:
        out["pred_churn_proba"] = pd.Series(y_proba, index=out.index)

    if y_true is not None:
        out["true_churn"] = y_true.values

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1       : {f1:.4f}")
        if y_proba is not None:
            auc = roc_auc_score(y_true, y_proba)
            print(f"  ROC-AUC  : {auc:.4f}")

    # Save output
    out_path = Path(output_csv)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / output_csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("✅ Prediction completed.")
    print(f"   Output : {out_path}")
    print(f"   Rows   : {out.shape[0]}")
    print(f"   Churn predicted (1) : {int(y_pred.sum())} / {len(y_pred)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run churn inference on new data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path (example: data/raw/new_clients.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/predictions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--target",
        default="Churn",
        help="Target column name if present in input",
    )
    parser.add_argument(
        "--date_format",
        default=None,
        help="Optional strptime format for RegistrationDate (example: %%d/%%m/%%Y)",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.target, args.date_format)