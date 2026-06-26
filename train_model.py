"""
train_model.py — Loan Approval Prediction
==========================================
AIML Summer Internship 2026 | MNNIT Allahabad

Run this script locally to retrain models and generate fresh .pkl files
that are compatible with your local Python/sklearn version.

Usage:
    python train_model.py

Output:
    Model/best_model.pkl
    Model/model_metrics.pkl
    Model/scaler.pkl
    Model/label_encoders.pkl
    Model/feature_names.pkl
    Model/metadata.json
"""

import os
import sys
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not found. Run: pip install xgboost")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("Model/training.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT   = Path(__file__).parent
DATA   = ROOT / "Dataset" / "loan_data.csv"
MODEL  = ROOT / "Model"
MODEL.mkdir(parents=True, exist_ok=True)

TRAIN_URLS = [
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/loan_data.csv",
    "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Loan%20Eligibility%20Prediction.csv",
]


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load dataset from local path or download from URL."""
    if DATA.exists():
        log.info(f"Loading local dataset: {DATA}")
        df = pd.read_csv(DATA)
    else:
        log.info("Local file not found. Downloading from URL...")
        df = None
        for url in TRAIN_URLS:
            try:
                df = pd.read_csv(url)
                log.info(f"Downloaded dataset from: {url}")
                break
            except Exception as e:
                log.warning(f"Dataset download failed for {url}: {e}")

        if df is None:
            log.info("Generating synthetic dataset for demonstration...")
            df = _generate_synthetic_data()

        DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA, index=False)

    log.info(f"Dataset shape: {df.shape}")
    return df


def _generate_synthetic_data(n=614, seed=42) -> pd.DataFrame:
    """Generate a realistic synthetic loan dataset."""
    rng = np.random.RandomState(seed)
    n_approved = int(n * 0.68)
    n_rejected = n - n_approved

    rows = []
    for i in range(n):
        approved = i < n_approved
        ch       = 1 if (rng.random() < 0.85 if approved else rng.random() < 0.2) else 0
        ai       = rng.randint(2500, 25000)
        ci       = rng.randint(0, 8000)
        la       = rng.randint(50, 500)
        rows.append({
            "Loan_ID"           : f"LP{100001+i}",
            "Gender"            : rng.choice(["Male","Female"], p=[0.8,0.2]),
            "Married"           : rng.choice(["Yes","No"],      p=[0.65,0.35]),
            "Dependents"        : rng.choice(["0","1","2","3+"], p=[0.57,0.17,0.16,0.10]),
            "Education"         : rng.choice(["Graduate","Not Graduate"], p=[0.78,0.22]),
            "Self_Employed"     : rng.choice(["No","Yes"], p=[0.85,0.15]),
            "ApplicantIncome"   : ai,
            "CoapplicantIncome" : ci,
            "LoanAmount"        : la,
            "Loan_Amount_Term"  : rng.choice([360,180,480,120,240,60,300,36,84],
                                              p=[0.70,0.10,0.06,0.04,0.04,0.02,0.02,0.01,0.01]),
            "Credit_History"    : ch,
            "Property_Area"     : rng.choice(["Urban","Semiurban","Rural"], p=[0.33,0.36,0.31]),
            "Loan_Status"       : "Y" if approved else "N",
        })
        # Inject some missing values
        for col, prob in [("Gender",0.02),("Married",0.01),("Dependents",0.02),
                          ("Self_Employed",0.03),("LoanAmount",0.04),
                          ("Credit_History",0.08),("Loan_Amount_Term",0.02)]:
            if rng.random() < prob:
                rows[-1][col] = None

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2. PREPROCESS + FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Full preprocessing + feature engineering pipeline."""
    data = df.copy()
    data.drop("Loan_ID", axis=1, inplace=True, errors="ignore")

    log.info("Missing values before imputation:")
    missing = data.isnull().sum()
    for col in missing[missing > 0].index:
        log.info(f"  {col}: {missing[col]}")

    # ── Impute ──
    cat_fill = ["Gender","Married","Dependents","Self_Employed","Credit_History","Loan_Amount_Term"]
    for col in cat_fill:
        if col in data.columns:
            mode = data[col].mode(dropna=True)
            if not mode.empty:
                data[col] = data[col].fillna(mode.iloc[0])

    data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())

    # ── Fix Dependents ──
    data["Dependents"] = (
        data["Dependents"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .replace("nan", np.nan)
    )
    data["Dependents"] = pd.to_numeric(data["Dependents"], errors="coerce")

    # ── Label Encode ──
    cat_cols = ["Gender","Married","Education","Self_Employed","Property_Area"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # ── Feature Engineering ──
    data["TotalIncome"]            = data["ApplicantIncome"] + data["CoapplicantIncome"]
    data["Log_ApplicantIncome"]    = np.log1p(data["ApplicantIncome"])
    data["Log_CoapplicantIncome"]  = np.log1p(data["CoapplicantIncome"])
    data["Log_LoanAmount"]         = np.log1p(data["LoanAmount"])
    data["Log_TotalIncome"]        = np.log1p(data["TotalIncome"])
    data["EMI"]                    = data["LoanAmount"] / data["Loan_Amount_Term"].replace(0, np.nan)
    data["EMI"] = data["EMI"].fillna(data["EMI"].median())
    data["Balance_Income"]         = data["TotalIncome"] - data["EMI"] * 1000
    data["Debt_Income_Ratio"]      = data["LoanAmount"] / data["TotalIncome"].replace(0, np.nan)
    data["Debt_Income_Ratio"] = data["Debt_Income_Ratio"].fillna(data["Debt_Income_Ratio"].median())

    data.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","TotalIncome"],
              axis=1, inplace=True)

    # Final NaN cleanup for pandas 3.x / coercion edge cases
    for col in data.columns:
        if data[col].isna().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                mode = data[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                data[col] = data[col].fillna(fill_value)

    # ── Encode Target ──
    data["Loan_Status"] = (data["Loan_Status"] == "Y").astype(int)

    print("\nNaNs after preprocessing:")
    print(data.isna().sum())
    print("Total:", data.isna().sum().sum())

    log.info(f"Preprocessing complete. Shape: {data.shape}")
    return data, label_encoders



# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train_evaluate(X_train, X_test, y_train, y_test,
                   X_train_scaled, X_test_scaled, feature_names):
    """Train all models and return results."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
        )

    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results  = {}
    trained  = {}

    for name, model in models.items():
        log.info(f"Training: {name}")
        X_tr = X_train_scaled if name == "Logistic Regression" else X_train.values
        X_te = X_test_scaled  if name == "Logistic Regression" else X_test.values
        import numpy as np

        print("\nTotal NaNs:", np.isnan(X_tr).sum())
        model.fit(X_tr, y_train)
        trained[name] = model

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        cv_acc = cross_val_score(model, X_tr, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

        metrics = {
            "Accuracy"  : round(accuracy_score(y_test, y_pred), 4),
            "Precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1_Score"  : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC_AUC"   : round(roc_auc_score(y_test, y_prob), 4),
            "CV_Mean"   : round(cv_acc.mean(), 4),
            "CV_Std"    : round(cv_acc.std(), 4),
        }
        results[name] = metrics

        log.info(
            f"  Acc={metrics['Accuracy']:.4f}  "
            f"Prec={metrics['Precision']:.4f}  "
            f"Rec={metrics['Recall']:.4f}  "
            f"F1={metrics['F1_Score']:.4f}  "
            f"AUC={metrics['ROC_AUC']:.4f}  "
            f"CV={metrics['CV_Mean']:.4f}±{metrics['CV_Std']:.4f}"
        )

    return results, trained


# ─────────────────────────────────────────────
# 4. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(trained, results, best_name, scaler, label_encoders, feature_names):
    """Persist all model artifacts."""
    # Best model
    joblib.dump(trained[best_name],  MODEL / "best_model.pkl")
    log.info("best_model.pkl saved")

    # Scaler
    joblib.dump(scaler,              MODEL / "scaler.pkl")
    log.info("scaler.pkl saved")

    # Label encoders
    joblib.dump(label_encoders,      MODEL / "label_encoders.pkl")
    log.info("label_encoders.pkl saved")

    # Feature names
    joblib.dump(feature_names,       MODEL / "feature_names.pkl")
    log.info("feature_names.pkl saved")

    # All model metrics
    model_metrics = {
        "results"         : results,
        "best_model_name" : best_name,
        "feature_names"   : feature_names,
        "python_version"  : sys.version,
        "sklearn_version" : __import__("sklearn").__version__,
    }
    joblib.dump(model_metrics,       MODEL / "model_metrics.pkl")
    log.info("model_metrics.pkl saved")

    # Individual models
    for name, model in trained.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL / f"{safe}.pkl")
        log.info(f"{safe}.pkl saved")

    # JSON metadata
    meta = {
        "best_model"    : best_name,
        "feature_names" : feature_names,
        "python_version": sys.version.split()[0],
        "metrics"       : results,
    }
    with open(MODEL / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("metadata.json saved")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Loan Approval Prediction - Training Pipeline")
    log.info(f"   Python  : {sys.version.split()[0]}")
    log.info(f"   sklearn : {__import__('sklearn').__version__}")
    log.info("=" * 60)

    # Load
    df = load_data()

    # Preprocess
    data, label_encoders = preprocess(df)

    # Split
    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train
    results, trained = train_evaluate(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, feature_names
    )

    # Best model
    best_name = max(results, key=lambda m: results[m]["ROC_AUC"])
    log.info(f"\nBest Model: {best_name}  (ROC-AUC = {results[best_name]['ROC_AUC']:.4f})")

    # Save
    save_artifacts(trained, results, best_name, scaler, label_encoders, feature_names)

    log.info("\n📊 FINAL RESULTS SUMMARY")
    log.info("-" * 60)
    log.info(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    log.info("-" * 60)
    for name, m in results.items():
        marker = " <- BEST" if name == best_name else ""
        log.info(
            f"{name:<25} {m['Accuracy']:>6.4f} {m['Precision']:>6.4f} "
            f"{m['Recall']:>6.4f} {m['F1_Score']:>6.4f} {m['ROC_AUC']:>6.4f}{marker}"
        )
    log.info("\nTraining complete! All artifacts saved to Model/")


if __name__ == "__main__":
    main()
