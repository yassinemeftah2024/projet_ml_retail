from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils import handle_outliers_iqr

app = Flask(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_TRAIN_TEST = PROJECT_ROOT / "data" / "train_test"

OUTLIER_COLS = ["SupportTicketsCount", "SatisfactionScore"]

model = joblib.load(MODELS_DIR / "best_model.joblib")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")


def load_outlier_bounds():
    path = MODELS_DIR / "outlier_bounds.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_dropped_features():
    path = REPORTS_DIR / "dropped_features.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("dropped_all", [])
    return []


def normalize_decimal_string(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    value = value.replace(" ", "")
    value = value.replace(",", ".")
    return value


def to_float(value: str):
    value = normalize_decimal_string(value)
    return pd.to_numeric(value, errors="coerce")


def prepare_input_df(form_data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Frequency": to_float(form_data["Frequency"]),
        "MonetaryTotal": to_float(form_data["MonetaryTotal"]),
        "CustomerTenureDays": to_float(form_data["CustomerTenureDays"]),
        "UniqueProducts": to_float(form_data["UniqueProducts"]),
        "ReturnRatio": to_float(form_data["ReturnRatio"]),
        "Age": to_float(form_data["Age"]),
        "SupportTicketsCount": to_float(form_data["SupportTicketsCount"]),
        "SatisfactionScore": to_float(form_data["SatisfactionScore"]),
        "Gender": str(form_data["Gender"]).strip(),
        "Country": str(form_data["Country"]).strip(),
    }])


def align_like_model(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    bounds = load_outlier_bounds()
    if bounds:
        X, _ = handle_outliers_iqr(X, OUTLIER_COLS, bounds=bounds)

    dropped = load_dropped_features()
    if dropped:
        X = X.drop(columns=dropped, errors="ignore")

    expected_cols = list(getattr(preprocessor, "feature_names_in_", []))
    if expected_cols:
        for c in expected_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[expected_cols]

    return X


def predict_from_df(X: pd.DataFrame):
    X_aligned = align_like_model(X)
    Xt = preprocessor.transform(X_aligned)

    pred = model.predict(Xt)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xt)[:, 1]

    return pred, proba


def load_test_row(row_index: int) -> pd.DataFrame:
    x_test_path = DATA_TRAIN_TEST / "X_test.csv"
    if not x_test_path.exists():
        raise FileNotFoundError("X_test.csv not found in data/train_test/")

    X_test = pd.read_csv(x_test_path)

    if row_index < 0 or row_index >= len(X_test):
        raise IndexError(f"Row index out of range. Must be between 0 and {len(X_test)-1}")

    return X_test.iloc[[row_index]].copy()


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    error = None
    mode = "manual"

    form_data = {
        "Frequency": "",
        "MonetaryTotal": "",
        "CustomerTenureDays": "",
        "UniqueProducts": "",
        "ReturnRatio": "",
        "Age": "",
        "SupportTicketsCount": "",
        "SatisfactionScore": "",
        "Gender": "",
        "Country": "",
        "row_index": "",
    }

    if request.method == "POST":
        action = request.form.get("action", "predict")
        mode = request.form.get("mode", "manual")

        if action == "reset":
            return render_template(
                "index.html",
                result=None,
                probability=None,
                error=None,
                form_data=form_data,
                mode=mode,
            )

        try:
            form_data = {
                "Frequency": request.form.get("Frequency", "").strip(),
                "MonetaryTotal": request.form.get("MonetaryTotal", "").strip(),
                "CustomerTenureDays": request.form.get("CustomerTenureDays", "").strip(),
                "UniqueProducts": request.form.get("UniqueProducts", "").strip(),
                "ReturnRatio": request.form.get("ReturnRatio", "").strip(),
                "Age": request.form.get("Age", "").strip(),
                "SupportTicketsCount": request.form.get("SupportTicketsCount", "").strip(),
                "SatisfactionScore": request.form.get("SatisfactionScore", "").strip(),
                "Gender": request.form.get("Gender", "").strip(),
                "Country": request.form.get("Country", "").strip(),
                "row_index": request.form.get("row_index", "").strip(),
            }

            if mode == "dataset":
                if form_data["row_index"] == "":
                    error = "Please enter a row index from X_test.csv."
                    return render_template(
                        "index.html",
                        result=None,
                        probability=None,
                        error=error,
                        form_data=form_data,
                        mode=mode,
                    )

                row_index = int(form_data["row_index"])
                df = load_test_row(row_index)

                # remplir automatiquement les 10 champs visibles pour affichage
                visible_cols = [
                    "Frequency", "MonetaryTotal", "CustomerTenureDays", "UniqueProducts",
                    "ReturnRatio", "Age", "SupportTicketsCount", "SatisfactionScore",
                    "Gender", "Country"
                ]
                for c in visible_cols:
                    if c in df.columns:
                        v = df.iloc[0][c]
                        form_data[c] = "" if pd.isna(v) else str(v)

                pred, proba = predict_from_df(df)

            else:
                required_manual = [
                    "Frequency", "MonetaryTotal", "CustomerTenureDays", "UniqueProducts",
                    "ReturnRatio", "Age", "SupportTicketsCount", "SatisfactionScore",
                    "Gender", "Country"
                ]
                missing_fields = [k for k in required_manual if form_data[k] == ""]
                if missing_fields:
                    error = "Please fill in all 10 fields before predicting."
                    return render_template(
                        "index.html",
                        result=None,
                        probability=None,
                        error=error,
                        form_data=form_data,
                        mode=mode,
                    )

                df = prepare_input_df(form_data)

                numeric_cols = [
                    "Frequency", "MonetaryTotal", "CustomerTenureDays", "UniqueProducts",
                    "ReturnRatio", "Age", "SupportTicketsCount", "SatisfactionScore"
                ]
                if df[numeric_cols].isna().any().any():
                    error = "Numeric fields must contain valid values. You can use either 0.20 or 0,20."
                    return render_template(
                        "index.html",
                        result=None,
                        probability=None,
                        error=error,
                        form_data=form_data,
                        mode=mode,
                    )

                pred, proba = predict_from_df(df)

            result = int(pred[0])
            if proba is not None:
                probability = float(proba[0])

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error,
        form_data=form_data,
        mode=mode,
    )


if __name__ == "__main__":
    app.run(debug=True)