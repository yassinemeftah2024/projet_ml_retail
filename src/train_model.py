from __future__ import annotations
from utils import save_corr_and_heatmap
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
# Use a non-interactive backend to avoid Tkinter GUI errors when running
# in headless or script environments (prevents RuntimeError at shutdown).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer


# ==========================
# Paths
# ==========================
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATA_TRAIN_TEST = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR      = PROJECT_ROOT / "models"
REPORTS_DIR     = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# Helpers
# ==========================
def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_dense(X):
    """Convert sparse matrix to dense if needed."""
    return X.toarray() if hasattr(X, "toarray") else X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Même logique que preprocessing.py pour cohérence."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    age_cols       = ["Age"] if "Age" in num_cols else []
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


# ==========================
# VIF
# ==========================
def compute_vif(X_train: pd.DataFrame) -> None:
    """VIF sur colonnes numériques (raw train, avant one-hot)."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        HAS_SM = True
    except Exception:
        HAS_SM = False

    status_path = REPORTS_DIR / "vif_status.txt"

    if not HAS_SM:
        status_path.write_text(
            "statsmodels not installed. Install with: pip install statsmodels",
            encoding="utf-8",
        )
        return

    num_df = X_train.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        status_path.write_text("Not enough numeric columns for VIF.", encoding="utf-8")
        return

    num_df = num_df.fillna(num_df.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(num_df.values)

    rows = []
    for i, col in enumerate(num_df.columns):
        rows.append({"feature": col, "vif": float(variance_inflation_factor(Xs, i))})

    pd.DataFrame(rows).sort_values("vif", ascending=False).to_csv(
        REPORTS_DIR / "vif_numeric.csv", index=False
    )
    status_path.write_text("VIF successfully computed.", encoding="utf-8")


# ==========================
# PCA  — FIX BUG 1
# ==========================
def run_pca_auto(
    preprocessor,
    X_train: pd.DataFrame,
    variance_threshold: float = 0.85,
    max_components: int | None = None,
) -> tuple[PCA, np.ndarray]:
    """
    PCA automatique:
    - choisit le nombre minimal de composantes pour atteindre variance_threshold
    - applique ensuite PCA finale avec ce nombre de composantes
    """
    Xt = _to_dense(preprocessor.transform(X_train))

    # PCA complète pour trouver combien de composantes garder
    pca_full = PCA(random_state=42)
    pca_full.fit(Xt)

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum >= variance_threshold) + 1)

    if max_components is not None:
        n_components = min(n_components, max_components)

    n_components = max(2, min(n_components, Xt.shape[1]))

    # PCA finale
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(Xt)

    evr = pca.explained_variance_ratio_
    cum_final = np.cumsum(evr)

    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum_final,
    }).to_csv(REPORTS_DIR / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cum_final) + 1), cum_final, marker="o")
    plt.axhline(y=variance_threshold, color="red", linestyle="--", label=f"target={variance_threshold}")
    plt.title("PCA - Variance cumulée expliquée (Train)")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance cumulée")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pca_cumulative_variance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(Z[:, 0], Z[:, 1], s=12)
    plt.title(f"PCA - Projection 2D (Train) | {n_components} composantes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pca_projection_2d.png", dpi=200)
    plt.close()

    joblib.dump(pca, MODELS_DIR / "pca.joblib")

    save_json(REPORTS_DIR / "pca_info.json", {
        "variance_threshold": variance_threshold,
        "selected_n_components": n_components,
        "final_cumulative_variance": float(cum_final[-1]),
    })

    print(f"  PCA auto : {n_components} composantes retenues | variance cumulée = {cum_final[-1]:.3f}")
    return pca, Z


# ==========================
# KMeans — FIX BUG 2
# ==========================
def run_kmeans_on_pca_improved(Z_train: np.ndarray, y_train: pd.Series) -> None:
    """
    KMeans amélioré :
    - teste k=2..10
    - compare silhouette, Davies-Bouldin, Calinski-Harabasz
    - choisit principalement par silhouette
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    ks = list(range(2, 11))
    inertias = []
    silhouettes = []
    db_scores = []
    ch_scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Z_train)

        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(Z_train, labels)))
        db_scores.append(float(davies_bouldin_score(Z_train, labels)))
        ch_scores.append(float(calinski_harabasz_score(Z_train, labels)))

    best_idx = int(np.argmax(silhouettes))
    best_k = ks[best_idx]

    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("KMeans - Elbow (Inertia) on TRAIN")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "kmeans_elbow.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ks, silhouettes, marker="o", label="Silhouette")
    plt.axvline(x=best_k, color="red", linestyle="--", alpha=0.6, label=f"best k={best_k}")
    plt.title("KMeans - Silhouette score on TRAIN")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "kmeans_silhouette.png", dpi=200)
    plt.close()

    pd.DataFrame({
        "k": ks,
        "inertia": inertias,
        "silhouette": silhouettes,
        "davies_bouldin": db_scores,
        "calinski_harabasz": ch_scores,
    }).to_csv(REPORTS_DIR / "kmeans_metrics.csv", index=False)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(Z_train)
    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")

    Z2 = Z_train[:, :2]
    plt.figure(figsize=(9, 5))
    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=14, cmap="tab10")
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"KMeans clusters on PCA map (k={best_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "kmeans_clusters_pca_map.png", dpi=200)
    plt.close()

    df_tmp = pd.DataFrame({"cluster": labels, "churn": y_train.values})
    summary = df_tmp.groupby("cluster").agg(
        count=("churn", "size"),
        churn_rate=("churn", "mean"),
    ).reset_index()
    summary.to_csv(REPORTS_DIR / "kmeans_cluster_summary.csv", index=False)

    save_json(REPORTS_DIR / "kmeans_results.json", {
        "best_k": best_k,
        "k_values": ks,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
        "davies_bouldin_scores": db_scores,
        "calinski_harabasz_scores": ch_scores,
        "n_pca_dims_used": int(Z_train.shape[1]),
    })
    

    print(f"✅ KMeans improved. Best k={best_k} on {Z_train.shape[1]} PCA dims.")
# ==========================
# Classification + GridSearch
# ==========================
def run_classification(
    preprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Train multiple classifiers, compare them by ROC-AUC then F1,
    and save the best one.

    Includes:
      - LogisticRegression baseline
      - LogisticRegression GridSearchCV
      - LogisticRegression Optuna (if optuna is installed)
      - SVM baseline
      - SVM GridSearchCV
      - RandomForest baseline
      - RandomForest GridSearchCV

    Saves:
      - reports/model_results.json
      - reports/confusion_matrix_best.png
      - reports/classification_report_best.txt
      - models/best_model.joblib
    """
    Xt = _to_dense(preprocessor.transform(X_train))
    Xs = _to_dense(preprocessor.transform(X_test))

    results = []
    best = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Logistic Regression baseline ───────────────────────────────────────
    lr = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(Xt, y_train)
    _eval_and_record(lr, Xt, Xs, y_train, y_test, "LogisticRegression", results)

    # ── Logistic Regression GridSearchCV ───────────────────────────────────
    print("  GridSearchCV sur LogisticRegression en cours…")
    param_grid_lr = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs"],
    }

    gs_lr = GridSearchCV(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        param_grid_lr,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    gs_lr.fit(Xt, y_train)
    best_lr = gs_lr.best_estimator_
    _eval_and_record(best_lr, Xt, Xs, y_train, y_test, "LogisticRegression_GridSearch", results)
    print(f"  LogisticRegression best params: {gs_lr.best_params_}")

    # ── Logistic Regression Optuna (optional) ──────────────────────────────
    best_lr_optuna = None
    lr_optuna_params = None
    try:
        import optuna
        from sklearn.model_selection import cross_val_score

        print("  Optuna sur LogisticRegression en cours…")

        def objective(trial):
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)

            model = LogisticRegression(
                C=C,
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            )

            scores = cross_val_score(
                model,
                Xt,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        lr_optuna_params = study.best_params

        best_lr_optuna = LogisticRegression(
            C=lr_optuna_params["C"],
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        best_lr_optuna.fit(Xt, y_train)
        _eval_and_record(
            best_lr_optuna,
            Xt,
            Xs,
            y_train,
            y_test,
            "LogisticRegression_Optuna",
            results,
        )
        print(f"  LogisticRegression Optuna best params: {lr_optuna_params}")

    except ImportError:
        print("  Optuna non installé, LogisticRegression_Optuna ignoré.")

    # ── SVM baseline ────────────────────────────────────────────────────────
    svm = SVC(
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(Xt, y_train)
    _eval_and_record(svm, Xt, Xs, y_train, y_test, "SVM", results)

    # ── SVM GridSearchCV ────────────────────────────────────────────────────
    print("  GridSearchCV sur SVM en cours…")
    param_grid_svm = {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto", 0.01, 0.001],
        "kernel": ["rbf"],
    }

    gs_svm = GridSearchCV(
        SVC(
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        param_grid_svm,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    gs_svm.fit(Xt, y_train)
    best_svm = gs_svm.best_estimator_
    _eval_and_record(best_svm, Xt, Xs, y_train, y_test, "SVM_GridSearch", results)
    print(f"  SVM best params: {gs_svm.best_params_}")

    # ── RandomForest baseline ───────────────────────────────────────────────
    rf_base = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_base.fit(Xt, y_train)
    _eval_and_record(rf_base, Xt, Xs, y_train, y_test, "RandomForest_baseline", results)

    # ── RandomForest GridSearchCV ───────────────────────────────────────────
    print("  GridSearchCV sur RandomForest en cours…")
    param_grid_rf = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5],
    }

    gs_rf = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        param_grid_rf,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    gs_rf.fit(Xt, y_train)
    best_rf = gs_rf.best_estimator_
    _eval_and_record(best_rf, Xt, Xs, y_train, y_test, "RandomForest_GridSearch", results)
    print(f"  RandomForest best params: {gs_rf.best_params_}")

    # ── Select best model globally ──────────────────────────────────────────
    for row in results:
        key = (
            row["roc_auc"] if row["roc_auc"] is not None else -1.0,
            row["f1"],
        )
        if best is None or key > (
            best["roc_auc"] if best["roc_auc"] is not None else -1.0,
            best["f1"],
        ):
            best = row.copy()

    model_map = {
        "LogisticRegression": lr,
        "LogisticRegression_GridSearch": best_lr,
        "SVM": svm,
        "SVM_GridSearch": best_svm,
        "RandomForest_baseline": rf_base,
        "RandomForest_GridSearch": best_rf,
    }

    if best_lr_optuna is not None:
        model_map["LogisticRegression_Optuna"] = best_lr_optuna

    best_clf = model_map[best["model"]]
    joblib.dump(best_clf, MODELS_DIR / "best_model.joblib")

    # ── Confusion matrix + classification report ────────────────────────────
    y_pred = best_clf.predict(Xs)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {best['model']}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix_best.png", dpi=200)
    plt.close()

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Fidèle (0)", "Churn (1)"],
    )
    (REPORTS_DIR / "classification_report_best.txt").write_text(
        f"Model: {best['model']}\n\n{report}",
        encoding="utf-8",
    )

    save_json(REPORTS_DIR / "model_results.json", {
        "all": results,
        "best": best,
        "gridsearch_params_lr": gs_lr.best_params_,
        "gridsearch_params_svm": gs_svm.best_params_,
        "gridsearch_params_rf": gs_rf.best_params_,
        "optuna_params_lr": lr_optuna_params,
    })

    print(f"✅ Classification done. Best model: {best['model']}")
    print(
        f"   Accuracy={best['accuracy']:.4f}  "
        f"F1={best['f1']:.4f}  "
        f"ROC-AUC={best['roc_auc']:.4f}"
    )
from sklearn.model_selection import StratifiedKFold, cross_validate

def run_classification_cv(preprocessor, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Validation croisée 5-fold pour la classification.
    Évalue la stabilité de LogisticRegression, SVM et RandomForest.
    """
    Xt = _to_dense(preprocessor.transform(X_train))

    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            Xt,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        rows.append({
            "model": name,
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_f1_std": float(scores["test_f1"].std()),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
        })

        print(
            f"  CV {name:20s} "
            f"acc={scores['test_accuracy'].mean():.4f}±{scores['test_accuracy'].std():.4f}  "
            f"f1={scores['test_f1'].mean():.4f}±{scores['test_f1'].std():.4f}  "
            f"auc={scores['test_roc_auc'].mean():.4f}±{scores['test_roc_auc'].std():.4f}"
        )

    pd.DataFrame(rows).to_csv(REPORTS_DIR / "classification_cv_results.csv", index=False)
    save_json(REPORTS_DIR / "classification_cv_results.json", {"results": rows})

def _eval_and_record(
    clf,
    Xt: np.ndarray,
    Xs: np.ndarray,
    y_train: pd.Series,
    y_test:  pd.Series,
    name:    str,
    results: list,
) -> None:
    """Évalue un classifieur et ajoute les métriques à results."""
    pred  = clf.predict(Xs)
    proba = clf.predict_proba(Xs)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = float(accuracy_score(y_test, pred))
    f1  = float(f1_score(y_test, pred, zero_division=0))
    auc = float(roc_auc_score(y_test, proba)) if proba is not None else None

    results.append({"model": name, "accuracy": acc, "f1": f1, "roc_auc": auc})
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"  {name:30s}  acc={acc:.4f}  f1={f1:.4f}  auc={auc_str}")


# ==========================
# Regression
# ==========================
def run_regression(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """
    Prédit MonetaryTotal à partir des autres features.
    Sauvegarde :
      - reports/regression_results.json
      - models/best_regressor.joblib
      - models/preprocessor_regression.joblib
    """
    target_reg = "MonetaryTotal"
    if target_reg not in X_train.columns or target_reg not in X_test.columns:
        save_json(REPORTS_DIR / "regression_results.json", {
            "error": f"'{target_reg}' not found in X_train/X_test, regression skipped."
        })
        print("⚠️  Regression skipped (MonetaryTotal not found).")
        return

    ytr = pd.to_numeric(X_train[target_reg], errors="coerce")
    yte = pd.to_numeric(X_test[target_reg],  errors="coerce")

    Xtr = X_train.drop(columns=[target_reg])
    Xte = X_test.drop( columns=[target_reg])

    pre_reg = build_preprocessor(Xtr)
    pre_reg.fit(Xtr)
    joblib.dump(pre_reg, MODELS_DIR / "preprocessor_regression.joblib")

    Xt = _to_dense(pre_reg.transform(Xtr))
    Xs = _to_dense(pre_reg.transform(Xte))

    regressors = {
        "Ridge":                Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
    }

    results = []
    best    = None

    for name, reg in regressors.items():
        reg.fit(Xt, ytr)
        pred = reg.predict(Xs)

        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae  = float(mean_absolute_error(yte, pred))
        r2   = float(r2_score(yte, pred))

        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"  {name:30s}  rmse={rmse:.2f}  mae={mae:.2f}  r2={r2:.4f}")

        if best is None or r2 > best["r2"]:
            best = {"model": name, "rmse": rmse, "mae": mae, "r2": r2}
            joblib.dump(reg, MODELS_DIR / "best_regressor.joblib")

    save_json(REPORTS_DIR / "regression_results.json", {"all": results, "best": best})
    print(f"✅ Regression done. Best: {best['model']} (R²={best['r2']:.4f})")

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error

def run_regression_cv(X_train: pd.DataFrame) -> None:
    """
    5-fold cross-validation for regression on MonetaryTotal.
    """
    target_reg = "MonetaryTotal"
    if target_reg not in X_train.columns:
        save_json(REPORTS_DIR / "regression_cv_results.json", {
            "error": f"'{target_reg}' not found in X_train, regression CV skipped."
        })
        print("⚠️  Regression CV skipped (MonetaryTotal not found).")
        return

    ytr = pd.to_numeric(X_train[target_reg], errors="coerce")
    Xtr = X_train.drop(columns=[target_reg])

    pre_reg = build_preprocessor(Xtr)
    Xt = _to_dense(pre_reg.fit_transform(Xtr))

    regressors = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def rmse_func(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    scoring = {
        "r2": "r2",
        "rmse": make_scorer(rmse_func, greater_is_better=False),
    }

    rows = []

    for name, model in regressors.items():
        scores = cross_validate(
            model,
            Xt,
            ytr,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        rmse_vals = -scores["test_rmse"]

        rows.append({
            "model": name,
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "cv_r2_std": float(scores["test_r2"].std()),
            "cv_rmse_mean": float(rmse_vals.mean()),
            "cv_rmse_std": float(rmse_vals.std()),
        })

        print(
            f"  CV {name:20s} "
            f"r2={scores['test_r2'].mean():.4f}±{scores['test_r2'].std():.4f}  "
            f"rmse={rmse_vals.mean():.2f}±{rmse_vals.std():.2f}"
        )

    pd.DataFrame(rows).to_csv(REPORTS_DIR / "regression_cv_results.csv", index=False)
    save_json(REPORTS_DIR / "regression_cv_results.json", {"results": rows})
def run_pca_course_visualization(preprocessor, X_train: pd.DataFrame) -> None:
    """
    PCA 2D pour visualisation pédagogique (comme dans le cours).
    Cette PCA sert uniquement à afficher les données dans un plan 2D.
    """
    Xt = _to_dense(preprocessor.transform(X_train))

    pca_2d = PCA(n_components=2, random_state=42)
    Z2 = pca_2d.fit_transform(Xt)

    evr = pca_2d.explained_variance_ratio_
    cum2 = np.cumsum(evr)

    pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum2,
    }).to_csv(REPORTS_DIR / "pca_2d_explained_variance.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z2[:, 0], Z2[:, 1], s=12, alpha=0.7)
    plt.title("PCA 2D - Visualisation des données (style cours)")
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pca_course_2d.png", dpi=200)
    plt.close()

    print(f"  PCA 2D (cours) : variance expliquée par PC1+PC2 = {cum2[-1]:.3f}")

# ==========================
# Main
# ==========================
def main():
    print("📂 Loading train/test splits…")
    X_train = pd.read_csv(DATA_TRAIN_TEST / "X_train.csv")
    X_test  = pd.read_csv(DATA_TRAIN_TEST / "X_test.csv")
    y_train = pd.read_csv(DATA_TRAIN_TEST / "y_train.csv").iloc[:, 0].astype(int)
    y_test  = pd.read_csv(DATA_TRAIN_TEST / "y_test.csv").iloc[:, 0].astype(int)

    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    print("\n1️⃣  Corrélation / Heatmap…")
    save_corr_and_heatmap(X_train, REPORTS_DIR, threshold=0.8)
    print("\n1️⃣  VIF…")
    compute_vif(X_train)

    print("\n2️⃣  PCA…")
    

    pca, Z = run_pca_auto(
        preprocessor,
        X_train,
        variance_threshold=0.85,
        max_components=50
    )
    print("\n3️⃣  KMeans clustering…")
    run_kmeans_on_pca_improved(Z, y_train)

    print("\n4️⃣  Classification…")
    run_classification(preprocessor, X_train, X_test, y_train, y_test)

    print("\n5️⃣  Regression…")
    run_regression(X_train, X_test)
    print("\n6️⃣  Classification CV…")
    run_classification_cv(preprocessor, X_train, y_train)

    print("\n7️⃣  Regression CV…")
    run_regression_cv(X_train)

    print("\n✅ Pipeline complet : VIF + PCA + KMeans + Classification + Régression.")


if __name__ == "__main__":
    main()