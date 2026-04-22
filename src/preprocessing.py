from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    add_feature_engineering,
    drop_high_correlation_numeric,
    drop_high_missing_columns,
    handle_outliers_iqr,
    ip_feature_engineering,
    parse_registration_date,
    save_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_TRAIN_TEST = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_TRAIN_TEST.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTLIER_COLS = ["SupportTicketsCount", "SatisfactionScore"]

SUSPECT_LEAKAGE_COLS = [
    "AccountStatus",
    "LoyaltyLevel",
    "SpendingCategory",
    "BasketSizeCategory",
    "AvgBasketValue",
]

FINAL_FEATURES = [
    # Block A
    "Frequency",
    "MonetaryTotal",
    "MonetaryAvg",
    "MonetaryStd",
    "AvgQuantityPerTransaction",
    "AvgDaysBetweenPurchases",
    "UniqueProducts",
    "AvgProductsPerTransaction",
    "UniqueCountries",
    "ProductDiversity",
    # Safe from B
    "PreferredDayOfWeek",
    "PreferredHour",
    "PreferredMonth",
    "WeekendPurchaseRatio",
    # Block C
    "NegativeQuantityCount",
    "ZeroPriceCount",
    "ReturnRatio",
    "SupportTicketsCount",
    "SatisfactionScore",
    # Static / demographic
    "Age",
    "AgeCategory",
    "FavoriteSeason",
    "PreferredTimeOfDay",
    "Region",
    "WeekendPreference",
    "Gender",
    "Country",
    "RegYear",
    "RegMonth",
    "RegDay",
    "RegWeekday",
    "IsPrivateIP",
    "IpFirstOctet",
]

REDUNDANT_COLS = [
    "CancelledTransactions",
    "UniqueInvoices",
    "UniqueDescriptions",
    "AvgLinesPerInvoice",
    "MonetaryMin",
    "MonetaryMax",
    "MinQuantity",
]

LEAKAGE_DROP_COLS = [
    "Recency",
    "TenureRatio",
    "MonetaryPerDay",
    "CustomerType",
    "RFMSegment",
    "AccountStatus",
    "LoyaltyLevel",
    "SpendingCategory",
    "BasketSizeCategory",
    "AvgBasketValue",
]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    age_cols = ["Age"] if "Age" in num_cols else []
    other_num_cols = [c for c in num_cols if c != "Age"]

    num_pipe_other = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    num_pipe_age = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if other_num_cols:
        transformers.append(("num_other", num_pipe_other, other_num_cols))
    if age_cols:
        transformers.append(("num_age", num_pipe_age, age_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers)


def audit_suspect_columns(df: pd.DataFrame, target: str, suspects: list[str]) -> None:
    print("\n" + "=" * 80)
    print("🔎 LEAKAGE AUDIT — suspicious columns")
    print("=" * 80)

    existing = [c for c in suspects if c in df.columns]
    missing = [c for c in suspects if c not in df.columns]

    print(f"Present suspicious columns: {existing}")
    if missing:
        print(f"Missing suspicious columns: {missing}")

    if target not in df.columns:
        print(f"⚠️ Target '{target}' not found. Audit skipped.")
        return

    for col in existing:
        print("\n" + "-" * 80)
        print(f"Column: {col}")
        print(f"dtype  : {df[col].dtype}")
        print(f"nunique: {df[col].nunique(dropna=False)}")
        print(f"NaN %  : {df[col].isna().mean() * 100:.2f}%")

        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                corr = df[[col, target]].corr(numeric_only=True).iloc[0, 1]
                print(f"Corr with {target}: {corr:.4f}")
                print(df[col].describe())
            else:
                print(pd.crosstab(df[col], df[target], normalize="index"))
        except Exception as exc:
            print(f"⚠️ Audit failed for {col}: {exc}")


def main(
    input_csv: str,
    target: str,
    test_size: float,
    random_state: int,
    date_format: str | None = None,
) -> None:
    print("🚀 MAIN STARTED", flush=True)

    df = pd.read_csv(DATA_RAW / input_csv)

    audit_suspect_columns(df, target=target, suspects=SUSPECT_LEAKAGE_COLS)

    if "NewsletterSubscribed" in df.columns:
        df = df.drop(columns=["NewsletterSubscribed"])

    if "ChurnRiskCategory" in df.columns:
        df = df.drop(columns=["ChurnRiskCategory"])

    for col in list(df.columns):
        if col != target and "churnrisk" in col.lower():
            df = df.drop(columns=[col])

    df = parse_registration_date(df, col="RegistrationDate", fmt=date_format)
    df = ip_feature_engineering(df, col="LastLoginIP")

    existing_redundant = [c for c in REDUNDANT_COLS if c in df.columns]
    if existing_redundant:
        df = df.drop(columns=existing_redundant)
        print(f"🧹 Dropped redundant columns: {existing_redundant}")

    existing_leakage = [c for c in LEAKAGE_DROP_COLS if c in df.columns]
    if existing_leakage:
        df = df.drop(columns=existing_leakage)

    df = add_feature_engineering(df)

    audit_suspect_columns(df, target=target, suspects=SUSPECT_LEAKAGE_COLS)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    existing_final = [c for c in FINAL_FEATURES if c in X.columns]
    missing_final = [c for c in FINAL_FEATURES if c not in X.columns]

    print("\n✅ FINAL FEATURE SET")
    print("Columns kept:")
    print(existing_final)

    if missing_final:
        print("Missing columns:")
        print(missing_final)

    X = X[existing_final].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    id_cols = [c for c in ("CustomerID",) if c in X_train.columns or c in X_test.columns]
    if id_cols:
        print(f"Dropping identifier columns: {id_cols}")
        X_train = X_train.drop(columns=id_cols, errors="ignore")
        X_test = X_test.drop(columns=id_cols, errors="ignore")

    present_outliers = [c for c in OUTLIER_COLS if c in X_train.columns]
    missing_outliers = [c for c in OUTLIER_COLS if c not in X_train.columns]

    if present_outliers:
        print(f"Applying IQR clipping on columns: {present_outliers}")
    if missing_outliers:
        print(f"Outlier columns not present in data (skipped): {missing_outliers}")

    X_train, outlier_bounds = handle_outliers_iqr(X_train, OUTLIER_COLS)
    X_test, _ = handle_outliers_iqr(X_test, OUTLIER_COLS, bounds=outlier_bounds)

    save_json(MODELS_DIR / "outlier_bounds.json", outlier_bounds)

    drops_missing = drop_high_missing_columns(X_train, threshold=0.5)
    drops_corr = drop_high_correlation_numeric(X_train, threshold=0.8)
    drops_all = sorted(set(drops_missing + drops_corr))

    X_train = X_train.drop(columns=drops_all, errors="ignore")
    X_test = X_test.drop(columns=drops_all, errors="ignore")

    save_json(
        REPORTS_DIR / "dropped_features.json",
        {
            "missing_gt_50": drops_missing,
            "high_corr_gt_0_8": drops_corr,
            "dropped_all": drops_all,
        },
    )

    print("Distribution y_train AVANT SMOTE :", y_train.value_counts().to_dict())

    try:
        from imblearn.over_sampling import SMOTE, SMOTENC
        from sklearn.preprocessing import OrdinalEncoder

        X_train_for_smote = X_train.copy()

        for col in X_train_for_smote.select_dtypes(include="number").columns:
            X_train_for_smote[col] = pd.to_numeric(X_train_for_smote[col], errors="coerce")
            X_train_for_smote[col] = X_train_for_smote[col].fillna(
                X_train_for_smote[col].median()
            )

        cat_cols_smote = X_train_for_smote.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        for col in cat_cols_smote:
            mode_vals = X_train_for_smote[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            X_train_for_smote[col] = X_train_for_smote[col].fillna(fill_value).astype(str)

        oe = None
        if cat_cols_smote:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train_for_smote[cat_cols_smote] = oe.fit_transform(
                X_train_for_smote[cat_cols_smote]
            )
            joblib.dump(oe, MODELS_DIR / "ordinal_encoder_smote.joblib")

            cat_indices = [X_train_for_smote.columns.get_loc(c) for c in cat_cols_smote]
            smote = SMOTENC(
                categorical_features=cat_indices,
                random_state=random_state,
                k_neighbors=5,
            )
            X_resampled_arr, y_resampled = smote.fit_resample(X_train_for_smote.values, y_train)
            X_resampled = pd.DataFrame(X_resampled_arr, columns=X_train_for_smote.columns)

            try:
                X_resampled[cat_cols_smote] = oe.inverse_transform(X_resampled[cat_cols_smote])
            except Exception:
                X_resampled[cat_cols_smote] = X_resampled[cat_cols_smote].astype(object)
        else:
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train_for_smote, y_train)
            X_resampled = pd.DataFrame(X_resampled, columns=X_train_for_smote.columns)

        X_train_final = X_resampled
        y_train_final = pd.Series(y_resampled.astype(int), name=target)

        print("Distribution y_train APRÈS resampling :", y_train_final.value_counts().to_dict())

        save_json(
            REPORTS_DIR / "smote_info.json",
            {
                "applied": True,
                "method": "SMOTENC" if cat_cols_smote else "SMOTE",
                "original_train_size": int(len(y_train)),
                "resampled_train_size": int(len(y_train_final)),
                "class_distribution_after": y_train_final.value_counts().to_dict(),
                "categorical_columns_for_smote": cat_cols_smote,
            },
        )

    except ImportError:
        print("⚠️ imbalanced-learn non installé. SMOTE ignoré.")
        X_train_final = X_train.copy()
        y_train_final = y_train.copy()
        save_json(
            REPORTS_DIR / "smote_info.json",
            {"applied": False, "reason": "imbalanced-learn not installed"},
        )

    X_train_final.to_csv(DATA_TRAIN_TEST / "X_train.csv", index=False)
    X_test.to_csv(DATA_TRAIN_TEST / "X_test.csv", index=False)
    y_train_final.to_frame(name=target).to_csv(DATA_TRAIN_TEST / "y_train.csv", index=False)
    y_test.to_frame(name=target).to_csv(DATA_TRAIN_TEST / "y_test.csv", index=False)

    preprocessor = build_preprocessor(X_train_final)
    preprocessor.fit(X_train_final)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print("✅ Preprocessing + SMOTE completed successfully.")
    print("Train shape (après SMOTE):", X_train_final.shape)
    print("Test  shape               :", X_test.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="retail_customers_COMPLETE_CATEGORICAL.csv")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--date_format",
        default=None,
        help="Optional strptime format for RegistrationDate (e.g. '%d/%m/%Y').",
    )

    args = parser.parse_args()
    main(args.input, args.target, args.test_size, args.random_state, args.date_format)