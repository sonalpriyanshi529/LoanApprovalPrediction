# Loan Approval Prediction System
## Technical Project Report
### AIML Summer Internship 2026 — MNNIT Allahabad

---

## 1. Problem Statement

Banks receive thousands of loan applications each day. The manual review process is slow, costly, and susceptible to human bias. This project builds a supervised Machine Learning classification system that predicts whether a given loan application should be approved or rejected, based on the applicant's demographic and financial profile.

**ML Task:** Binary Classification  
**Target Variable:** `Loan_Status` — Approved (Y=1) or Rejected (N=0)

---

## 2. Dataset Description

**Source:** Kaggle — Loan Prediction Problem Dataset  
**Records:** 614 applications  
**Features:** 12 input features + 1 target  

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Ordinal | 0, 1, 2, 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Continuous | Monthly income (₹) |
| CoapplicantIncome | Continuous | Co-applicant monthly income (₹) |
| LoanAmount | Continuous | Loan amount in thousands (₹) |
| Loan_Amount_Term | Continuous | Repayment term in months |
| Credit_History | Binary | 1 = Meets guidelines, 0 = Does not |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| **Loan_Status** | **Target** | **Y = Approved, N = Rejected** |

**Class Distribution:** ~68% Approved, ~32% Rejected

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Missing Values
| Column | Missing Count | Imputation Strategy |
|--------|:---:|---|
| Gender | 13 | Mode |
| Married | 3 | Mode |
| Dependents | 15 | Mode |
| Self_Employed | 32 | Mode |
| LoanAmount | 22 | Median |
| Loan_Amount_Term | 14 | Mode |
| Credit_History | 50 | Mode |

### 3.2 Key EDA Findings

- **Credit History** is the single strongest predictor. ~80% of applicants with good credit history were approved vs ~8% without.
- **Property Area**: Semiurban applicants had the highest approval rate (~76%).
- **Education**: Graduates showed higher approval rates than non-graduates.
- **Income Distribution**: Highly right-skewed; log-transformation was applied.
- **Gender**: Males dominated the dataset (~80%) but gender alone showed little predictive power.

---

## 4. Data Preprocessing

1. **Drop** `Loan_ID` (identifier, not predictive)
2. **Impute** missing categorical values with mode; `LoanAmount` with median
3. **Fix** `Dependents`: convert "3+" → 3 (numeric)
4. **Encode** categorical features with `LabelEncoder`
5. **Scale** numerical features with `StandardScaler` (for Logistic Regression only)

---

## 5. Feature Engineering

Five additional features were derived to capture financial health:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `Log_ApplicantIncome` | log(1 + ApplicantIncome) | Reduce right skew |
| `Log_CoapplicantIncome` | log(1 + CoapplicantIncome) | Reduce right skew |
| `Log_LoanAmount` | log(1 + LoanAmount) | Reduce right skew |
| `Log_TotalIncome` | log(1 + ApplicantIncome + CoapplicantIncome) | Combined earning power |
| `EMI` | LoanAmount / Loan_Amount_Term | Monthly repayment burden |
| `Balance_Income` | TotalIncome − EMI×1000 | Disposable income after EMI |
| `Debt_Income_Ratio` | LoanAmount / TotalIncome | Debt load relative to income |

The original raw income and loan columns were dropped after log-transformation.

**Final Feature Count:** 15 features

---

## 6. Model Building

### 6.1 Train-Test Split
- **Split ratio:** 80% train / 20% test
- **Stratification:** Yes (preserve class balance)
- **Random seed:** 42

### 6.2 Models & Hyperparameters

**Logistic Regression**
```
C=1.0, max_iter=1000, solver='lbfgs', random_state=42
Input: StandardScaler-normalised features
```

**Random Forest**
```
n_estimators=200, max_depth=10, min_samples_split=5
n_jobs=-1, random_state=42
Input: Raw (unscaled) features
```

**XGBoost**
```
n_estimators=200, max_depth=6, learning_rate=0.1
subsample=0.8, colsample_bytree=0.8
eval_metric='logloss', random_state=42
Input: Raw (unscaled) features
```

### 6.3 Cross-Validation
- **Strategy:** StratifiedKFold (k=5)
- **Metric:** Accuracy
- Reported as Mean ± Std

---

## 7. Model Evaluation

### 7.1 Metrics Used

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Cost of false approvals |
| Recall | TP/(TP+FN) | Cost of missed approvals |
| F1-Score | 2×(P×R)/(P+R) | Balance of precision & recall |
| ROC-AUC | Area under ROC curve | Rank discrimination ability |

### 7.2 Results Summary (typical)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | ~80% | ~80% | ~93% | ~86% | ~82% |
| Random Forest | ~78% | ~79% | ~90% | ~84% | ~80% |
| **XGBoost** | **~82%** | **~83%** | **~92%** | **~87%** | **~85%** |

*Exact values depend on the dataset version and random seed.*

**Best Model:** XGBoost (highest ROC-AUC)

### 7.3 Model Selection Rationale
ROC-AUC was chosen as the primary selection criterion because:
- It is threshold-independent
- It measures the model's ability to rank positive cases above negative ones
- It is robust to class imbalance

---

## 8. Deployment

### 8.1 Saved Artifacts
| File | Contents |
|------|---------|
| `best_model.pkl` | Serialised best model |
| `model_metrics.pkl` | All evaluation results |
| `scaler.pkl` | Fitted StandardScaler |
| `label_encoders.pkl` | Fitted LabelEncoders |
| `feature_names.pkl` | Ordered feature list |
| `metadata.json` | Human-readable summary |

### 8.2 Streamlit Application
The `Streamlit_App/app.py` provides:
- Form-based input for all 11 applicant features
- Real-time prediction with confidence score
- Key risk factor analysis
- Model performance dashboard
- Multi-model comparison table
- Version-mismatch handling with clear fix instructions

### 8.3 Version Mismatch Handling
If Colab-trained `.pkl` files fail locally due to pickle protocol differences, the user can run `python train_model.py` to regenerate all artifacts in the local environment. The app displays a clear error message and instructions when this occurs.

---

## 9. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates .pkl files)
python train_model.py

# 3. Launch app
streamlit run Streamlit_App/app.py
```

---

## 10. Key Learnings

1. **Credit History** dominates — a single binary feature can be more powerful than many continuous features.
2. **Log transformations** significantly improved model performance on skewed income/loan features.
3. **Feature Engineering** (EMI, Balance_Income, Debt Ratio) added measurable signal.
4. **XGBoost** consistently outperformed both linear and random forest baselines on this dataset.
5. **Pickle versioning** is a practical deployment concern: always regenerate `.pkl` files in the target environment.

---

## 11. Future Work

- Hyperparameter optimisation with `GridSearchCV` or `Optuna`
- SHAP values for individual prediction explainability
- Data collection to address class imbalance (SMOTE)
- Real-time model monitoring dashboard
- REST API endpoint using FastAPI for integration with banking systems

---

*Report prepared as part of AIML Summer Internship 2026, MNNIT Allahabad.*
