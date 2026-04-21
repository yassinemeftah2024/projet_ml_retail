from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

from utils import (
    parse_registration_date,
    ip_feature_engineering,
    handle_outliers_iqr,       # FIX : signature mise à jour (retourne aussi les bornes)
    add_feature_engineering,
    drop_high_missing_columns,
    drop_high_correlation_numeric,
    save_json,
)

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_TRAIN_TEST = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR      = PROJECT_ROOT / "models"
REPORTS_DIR     = PROJECT_ROOT / "reports"

DATA_TRAIN_TEST.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Colonnes concernées par le clipping IQR (noms réels dans le dataset)
# Note: les noms doivent correspondre aux colonnes du CSV d'entrée.
OUTLIER_COLS = ["SupportTicketsCount", "SatisfactionScore"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    age_cols      = ["Age"] if "Age" in num_cols else []
    other_num_cols = [c for c in num_cols if c != "Age"]

    num_pipe_other = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    num_pipe_age = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler",  StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if other_num_cols:
        transformers.append(("num_other", num_pipe_other, other_num_cols))
    if age_cols:
        transformers.append(("num_age", num_pipe_age, age_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers)


def main(input_csv: str, target: str, test_size: float, random_state: int, date_format: str | None = None):
    df = pd.read_csv(DATA_RAW / input_csv)

    # ── Suppression features inutiles / fuite d'info ───────────────────────
    if "NewsletterSubscribed" in df.columns:       # variance nulle
        df = df.drop(columns=["NewsletterSubscribed"])
    if "ChurnRiskCategory" in df.columns:          # dérivée du churn → leakage
        df = df.drop(columns=["ChurnRiskCategory"])
    for c in list(df.columns):
        if c != target and "churnrisk" in c.lower():
            df = df.drop(columns=[c])

    # ── Parsing + feature engineering ──────────────────────────────────────
    df = parse_registration_date(df, col="RegistrationDate", fmt=date_format)
    df = ip_feature_engineering(df, col="LastLoginIP")
    # ── DROP REDUNDANT FEATURES (analysis-based) ─────────────
    REDUNDANT_COLS = [
        "CancelledTransactions",
        "UniqueInvoices",
        "UniqueDescriptions",
        "AvgLinesPerInvoice",
        "MonetaryMin",
        "MonetaryMax",
        "MinQuantity",
    ]

    existing_cols = [c for c in REDUNDANT_COLS if c in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"🧹 Dropped redundant columns: {existing_cols}")

    # Remove Recency and features derived from it to prevent target leakage.
    # If `Churn` was computed from Recency (as in the raw data), keeping
    # Recency or features derived from it lets the model learn a tautology.
    leak_cols = ["Recency", "TenureRatio", "MonetaryPerDay", "CustomerType", "RFMSegment"]
    for c in leak_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Feature engineering (MonetaryPerDay / AvgBasketValue / TenureRatio)
    # will now run without `Recency` present, so derived columns won't be created.
    df = add_feature_engineering(df)

    # ── Split X / y  (y jamais normalisé) ──────────────────────────────────
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # ── Train / Test split stratifié ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ── FIX: drop identifier columns (no predictive value, causes leakage if kept)
    id_cols = [c for c in ("CustomerID",) if c in X_train.columns or c in X_test.columns]
    if id_cols:
        print(f"Dropping identifier columns before SMOTE/saving splits: {id_cols}")
        X_train = X_train.drop(columns=id_cols, errors="ignore")
        X_test = X_test.drop(columns=id_cols, errors="ignore")

    # ── FIX BUG 3 : handle_outliers_iqr sur train UNIQUEMENT ───────────────
    # Les bornes IQR sont calculées sur X_train, puis appliquées sur X_test
    # pour éviter le data leakage.
    present = [c for c in OUTLIER_COLS if c in X_train.columns]
    missing = [c for c in OUTLIER_COLS if c not in X_train.columns]
    if present:
        print(f"Applying IQR clipping on columns: {present}")
    if missing:
        print(f"Outlier columns not present in data (skipped): {missing}")

    X_train, outlier_bounds = handle_outliers_iqr(X_train, OUTLIER_COLS)
    X_test, _ = handle_outliers_iqr(X_test, OUTLIER_COLS, bounds=outlier_bounds)

    # Sauvegarde des bornes pour les réutiliser à l'inférence (predict.py)
    save_json(MODELS_DIR / "outlier_bounds.json", outlier_bounds)

    # ── Suppression features inutiles (décisions sur train seulement) ──────
    drops_missing = drop_high_missing_columns(X_train, threshold=0.5)
    drops_corr = drop_high_correlation_numeric(X_train, threshold=0.8)
    drops_all = sorted(set(drops_missing + drops_corr))

    X_train = X_train.drop(columns=drops_all, errors="ignore")
    X_test = X_test.drop(columns=drops_all, errors="ignore")

    save_json(REPORTS_DIR / "dropped_features.json", {
        "missing_gt_50": drops_missing,
        "high_corr_gt_0_8": drops_corr,
        "dropped_all": drops_all,
    })

    # ── Affichage distribution des classes avant SMOTE ─────────────────────
    print("Distribution y_train AVANT SMOTE :", y_train.value_counts().to_dict())

    # ── FIX SMOTE : rééquilibrage du déséquilibre de classes ───────────────
    # SMOTE appliqué UNIQUEMENT sur le train.
    try:
        from imblearn.over_sampling import SMOTENC, SMOTE
        from sklearn.preprocessing import OrdinalEncoder

        # Imputation manuelle des NaN avant resampling
        X_train_for_smote = X_train.copy()

        # 1) Numériques -> médiane
        for col in X_train_for_smote.select_dtypes(include="number").columns:
            X_train_for_smote[col] = pd.to_numeric(X_train_for_smote[col], errors="coerce")
            X_train_for_smote[col] = X_train_for_smote[col].fillna(
                X_train_for_smote[col].median()
            )

        # 2) Catégorielles -> mode
        cat_cols_smote = X_train_for_smote.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in cat_cols_smote:
            mode_vals = X_train_for_smote[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            X_train_for_smote[col] = X_train_for_smote[col].fillna(fill_value).astype(str)

        # 3) Encodage ordinal temporaire
        oe = None
        if cat_cols_smote:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train_for_smote[cat_cols_smote] = oe.fit_transform(
                X_train_for_smote[cat_cols_smote]
            )
            joblib.dump(oe, MODELS_DIR / "ordinal_encoder_smote.joblib")

        # 4) Resampling
        if cat_cols_smote:
            cat_indices = [X_train_for_smote.columns.get_loc(c) for c in cat_cols_smote]
            smote = SMOTENC(
                categorical_features=cat_indices,
                random_state=random_state,
                k_neighbors=5
            )
            X_resampled_arr, y_resampled = smote.fit_resample(X_train_for_smote.values, y_train)
            X_resampled = pd.DataFrame(X_resampled_arr, columns=X_train_for_smote.columns)

            # Remettre les catégorielles en labels originaux
            try:
                X_resampled[cat_cols_smote] = oe.inverse_transform(
                    X_resampled[cat_cols_smote]
                )
            except Exception:
                X_resampled[cat_cols_smote] = X_resampled[cat_cols_smote].astype(object)

        else:
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train_for_smote, y_train)
            X_resampled = pd.DataFrame(X_resampled, columns=X_train_for_smote.columns)

        y_train_final = pd.Series(y_resampled.astype(int), name=target)
        X_train_final = X_resampled

        print("Distribution y_train APRÈS resampling :", y_train_final.value_counts().to_dict())
        save_json(REPORTS_DIR / "smote_info.json", {
            "applied": True,
            "method": "SMOTENC" if cat_cols_smote else "SMOTE",
            "original_train_size": int(len(y_train)),
            "resampled_train_size": int(len(y_train_final)),
            "class_distribution_after": y_train_final.value_counts().to_dict(),
            "categorical_columns_for_smote": cat_cols_smote,
        })

    except ImportError:
        print("⚠️  imbalanced-learn non installé. SMOTE/SMOTENC ignoré.")
        print("   Installer avec : pip install imbalanced-learn")
        X_train_final = X_train.copy()
        y_train_final = y_train.copy()
        save_json(REPORTS_DIR / "smote_info.json", {"applied": False, "reason": "imbalanced-learn not installed"})

    # ── Sauvegarde des splits ───────────────────────────────────────────────
    # Attention : X_test reste inchangé (jamais de SMOTE sur le test !)
    X_train_final.to_csv(DATA_TRAIN_TEST / "X_train.csv", index=False)
    X_test.to_csv(DATA_TRAIN_TEST / "X_test.csv", index=False)
    y_train_final.to_frame(name=target).to_csv(DATA_TRAIN_TEST / "y_train.csv", index=False)
    y_test.to_frame(name=target).to_csv(DATA_TRAIN_TEST / "y_test.csv", index=False)

    # ── Fit preprocessor sur train rééquilibré (no leakage) ────────────────
    preprocessor = build_preprocessor(X_train_final)
    preprocessor.fit(X_train_final)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print("✅ Preprocessing + SMOTE completed successfully.")
    print("Train shape (après SMOTE):", X_train_final.shape)
    print("Test  shape               :", X_test.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default="retail_customers_COMPLETE_CATEGORICAL.csv")
    parser.add_argument("--target",       default="Churn")
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument(
        "--date_format",
        default=None,
        help="Optional strptime format for RegistrationDate (e.g. '%d/%m/%Y').",
    )
    args = parser.parse_args()

    main(args.input, args.target, args.test_size, args.random_state, args.date_format)