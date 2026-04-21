from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# =========================
# Helpers IO
# =========================

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =========================
# Parsing & Cleaning
# =========================

def parse_registration_date(
    df: pd.DataFrame,
    col: str = "RegistrationDate",
    fmt: str | None = None,
) -> pd.DataFrame:
    """
    Parsing RegistrationDate + extraction (RegYear/RegMonth/RegDay/RegWeekday).

    Parameters
    - df: DataFrame containing the date column.
    - col: name of the date column to parse.
    - fmt: optional datetime format string to pass to `pd.to_datetime`.

    If `fmt` is provided, it will be used to parse the column deterministically
    which avoids fallback parsing warnings. The raw date column is dropped.
    """
    df = df.copy()
    if col not in df.columns:
        return df

    if fmt:
        dt = pd.to_datetime(df[col], format=fmt, dayfirst=True, errors="coerce")
    else:
        # Let pandas infer format (falls back to dateutil if ambiguous).
        # Note: some pandas versions removed `infer_datetime_format` keyword,
        # so call without it for broader compatibility.
        dt = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    df["RegYear"] = dt.dt.year
    df["RegMonth"] = dt.dt.month
    df["RegDay"] = dt.dt.day
    df["RegWeekday"] = dt.dt.weekday

    df = df.drop(columns=[col])
    return df


def ip_feature_engineering(df: pd.DataFrame, col: str = "LastLoginIP") -> pd.DataFrame:
    """Detect private IP + extract first octet, then drop raw IP column."""
    df = df.copy()
    if col not in df.columns:
        return df

    def _is_private(x):
        try:
            return float(ipaddress.ip_address(str(x)).is_private)
        except Exception:
            return np.nan

    def _first_octet(x):
        try:
            parts = str(x).split(".")
            if len(parts) == 4:
                return float(int(parts[0]))
            return np.nan
        except Exception:
            return np.nan

    df["IsPrivateIP"]  = df[col].apply(_is_private)
    df["IpFirstOctet"] = df[col].apply(_first_octet)

    df = df.drop(columns=[col])
    return df


def handle_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    bounds: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Clip outliers using IQR rule for selected numeric columns.

    - Si bounds est None  → calcul des bornes à partir de df (usage train).
    - Si bounds est fourni → applique les bornes pré-calculées (usage test/inférence).

    Retourne (df_clipped, bounds_dict) pour permettre la réutilisation sur test/inférence.
    """
    df = df.copy()
    computed_bounds: dict = {}

    for col in columns:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        if bounds is not None and col in bounds:
            lower, upper = bounds[col]["lower"], bounds[col]["upper"]
        else:
            q1  = s.quantile(0.25)
            q3  = s.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                df[col] = s
                computed_bounds[col] = {"lower": float(s.min()), "upper": float(s.max())}
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        df[col] = s.clip(lower, upper)
        computed_bounds[col] = {"lower": float(lower), "upper": float(upper)}

    return df, computed_bounds


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering :
    - MonetaryPerDay
    - AvgBasketValue
    - TenureRatio
    """
    df = df.copy()

    if {"MonetaryTotal", "Recency"}.issubset(df.columns):
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

    if {"MonetaryTotal", "Frequency"}.issubset(df.columns):
        freq = df["Frequency"].replace(0, np.nan)
        df["AvgBasketValue"] = df["MonetaryTotal"] / freq

    if {"Recency", "CustomerTenureDays"}.issubset(df.columns):
        ten = df["CustomerTenureDays"].replace(0, np.nan)
        df["TenureRatio"] = df["Recency"] / ten
    elif {"Recency", "CustomerTenure"}.issubset(df.columns):
        ten = df["CustomerTenure"].replace(0, np.nan)
        df["TenureRatio"] = df["Recency"] / ten

    return df


# =========================
# Feature dropping (train only)
# =========================

def drop_high_missing_columns(X_train: pd.DataFrame, threshold: float = 0.5) -> List[str]:
    miss_ratio = X_train.isna().mean()
    return miss_ratio[miss_ratio > threshold].index.tolist()


def drop_high_correlation_numeric(X_train: pd.DataFrame, threshold: float = 0.8) -> List[str]:
    num_df = X_train.select_dtypes(exclude=["object", "string"]).copy()
    if num_df.shape[1] <= 1:
        return []
    corr  = num_df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col] > threshold)]


# =========================
# Reports : Corr / VIF / PCA
# =========================

def save_corr_and_heatmap(
    X_train: pd.DataFrame,
    reports_dir: Path,
    threshold: float = 0.8,
) -> List[str]:
    """
    Corrélation + heatmap (numériques only) sur TRAIN.
    Sauvegarde :
      - correlation_matrix_numeric.csv
      - correlation_heatmap_numeric.png
      - high_corr_pairs_gt_0_8.csv
    Retourne : liste des features candidates à supprimer (corr > threshold).
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    num_df = X_train.select_dtypes(include=np.number)
    corr   = num_df.corr().fillna(0.0)
    corr.to_csv(reports_dir / "correlation_matrix_numeric.csv", index=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr.values, aspect="auto")
    plt.title("Heatmap Corrélation - Features Numériques (Train)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)),   corr.index,   fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(reports_dir / "correlation_heatmap_numeric.png", dpi=200)
    plt.close()

    abs_corr = corr.abs()
    upper    = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        s = upper[col][upper[col] > threshold]
        for row, val in s.items():
            pairs.append((row, col, float(val)))

    pd.DataFrame(pairs, columns=["feature_1", "feature_2", "abs_corr"]).sort_values(
        "abs_corr", ascending=False
    ).to_csv(reports_dir / "high_corr_pairs_gt_0_8.csv", index=False)

    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return to_drop


def compute_vif_numeric(X_train: pd.DataFrame, reports_dir: Path) -> None:
    """
    VIF > 10 = multicolinéarité sévère.
    Sauvegarde : vif_numeric.csv
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_STATSMODELS:
        (reports_dir / "vif_status.txt").write_text(
            "statsmodels not installed. Install with: pip install statsmodels",
            encoding="utf-8",
        )
        return

    num_df = X_train.select_dtypes(include=np.number).copy()
    num_df = num_df.fillna(num_df.median(numeric_only=True))

    Xs = StandardScaler().fit_transform(num_df)

    rows = []
    for i, col in enumerate(num_df.columns):
        rows.append({"feature": col, "vif": float(variance_inflation_factor(Xs, i))})

    pd.DataFrame(rows).sort_values("vif", ascending=False).to_csv(
        reports_dir / "vif_numeric.csv", index=False
    )


def run_pca_on_preprocessed(
    preprocessor,
    X_train: pd.DataFrame,
    reports_dir: Path,
    models_dir: Path,
    n_components: int = 10,
) -> Tuple[PCA, np.ndarray]:
    """
    PCA sur features préprocessées (onehot + imputation + scaling).
    Sauvegarde :
      - pca_explained_variance.csv
      - pca_cumulative_variance.png
      - pca_projection_2d.png
      - models/pca.joblib

    Retourne (pca, Z) où Z est la projection complète (toutes composantes).
    """
    import joblib

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    Xt = preprocessor.transform(X_train)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    # FIX : on utilise bien le paramètre n_components passé en argument
    n_components = max(2, min(int(n_components), Xt.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    Z   = pca.fit_transform(Xt)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum,
    }).to_csv(reports_dir / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cum) + 1), cum, marker="o")
    plt.title("PCA - Variance cumulée expliquée (Train)")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance cumulée")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / "pca_cumulative_variance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(Z[:, 0], Z[:, 1], s=12)
    plt.title("PCA - Projection 2D (Train)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(reports_dir / "pca_projection_2d.png", dpi=200)
    plt.close()

    joblib.dump(pca, models_dir / "pca.joblib")

    # Retourner Z complet (toutes les composantes) pour usage en clustering
    return pca, Z