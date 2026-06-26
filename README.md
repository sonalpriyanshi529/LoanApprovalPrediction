# 🏦 Loan Approval Prediction System
### AIML Summer Internship 2026 — MNNIT Allahabad

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)

---

## 📌 Project Overview

An end-to-end Machine Learning system that predicts loan approval outcomes for bank applicants. The system follows the complete ML lifecycle — from raw data ingestion through exploratory analysis, feature engineering, multi-model training, and evaluation — to a production-ready Streamlit web application.

**Task:** Binary Classification → Loan Approved (Y) / Rejected (N)  
**Dataset:** [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) (614 records, 13 features)

---

## 🗂️ Project Structure

```
LoanApprovalPrediction/
│
├── Dataset/
│   └── loan_data.csv                  ← Raw dataset (download from Kaggle)
│
├── Notebook/
│   ├── Loan_Approval_Prediction.ipynb ← Complete Google Colab notebook
│   ├── eda_categorical.png            ← EDA: categorical charts
│   ├── eda_numerical.png              ← EDA: numerical distributions
│   ├── eda_correlation.png            ← EDA: correlation heatmap
│   ├── model_comparison.png           ← Model performance bars
│   ├── confusion_matrices.png         ← All confusion matrices
│   ├── roc_curves.png                 ← ROC-AUC curves
│   └── feature_importance.png         ← Feature importance chart
│
├── Model/
│   ├── best_model.pkl                 ← Best performing model
│   ├── model_metrics.pkl              ← All evaluation results
│   ├── scaler.pkl                     ← StandardScaler
│   ├── label_encoders.pkl             ← LabelEncoders for categorical features
│   ├── feature_names.pkl              ← Ordered list of model features
│   ├── metadata.json                  ← Human-readable metadata
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── Streamlit_App/
│   └── app.py                         ← Production Streamlit application
│
├── Documentation/
│   └── Project_Report.pdf             ← Internship project report
│
├── train_model.py                     ← Local retraining script
├── requirements.txt                   ← Python dependencies
└── README.md                          ← This file
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project

```bash
git clone https://github.com/yourusername/LoanApprovalPrediction.git
cd LoanApprovalPrediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

**Option A — Kaggle API (recommended)**

```bash
# Place your kaggle.json in ~/.kaggle/ then:
kaggle datasets download -d altruistdelhite04/loan-prediction-problem-dataset
unzip loan-prediction-problem-dataset.zip -d Dataset/
mv Dataset/train.csv Dataset/loan_data.csv
```

**Option B — Manual download**

1. Go to https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
2. Download `train.csv` → rename to `loan_data.csv` → place in `Dataset/`

**Option C — Auto-download via script**

The `train_model.py` script will attempt to download the dataset automatically if `Dataset/loan_data.csv` is not found.

---

## 🚀 Quick Start

### Step 1 — Train models locally

```bash
python train_model.py
```

This generates all `.pkl` files in `Model/`, ensuring version compatibility with your local Python environment.

### Step 2 — Launch the Streamlit app

```bash
streamlit run Streamlit_App/app.py
```

Open your browser at http://localhost:8501

---

## 📓 Google Colab Notebook

1. Open `Notebook/Loan_Approval_Prediction.ipynb` in Google Colab
2. Run all cells (`Runtime → Run all`)
3. Download `Model_artifacts.zip` when prompted
4. Extract to your local `Model/` directory
5. If you see a version mismatch error, run `python train_model.py` instead

---

## 🤖 Models Trained

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear baseline; uses StandardScaler |
| **Random Forest** | Ensemble of 200 decision trees |
| **XGBoost** | Gradient boosting; typically best performer |

**Evaluation Metrics:** Accuracy · Precision · Recall · F1-Score · ROC-AUC · 5-Fold CV

---

## 🔧 Feature Engineering

| Feature | Formula |
|---------|---------|
| `Log_ApplicantIncome` | `log1p(ApplicantIncome)` |
| `Log_CoapplicantIncome` | `log1p(CoapplicantIncome)` |
| `Log_LoanAmount` | `log1p(LoanAmount)` |
| `Log_TotalIncome` | `log1p(ApplicantIncome + CoapplicantIncome)` |
| `EMI` | `LoanAmount / Loan_Amount_Term` |
| `Balance_Income` | `TotalIncome − (EMI × 1000)` |
| `Debt_Income_Ratio` | `LoanAmount / TotalIncome` |

---

## 🛠️ Troubleshooting

### `ValueError: Protocol 5 is not supported` or sklearn version error

```bash
# Regenerate .pkl files with YOUR local Python/sklearn version:
python train_model.py
```

### `ModuleNotFoundError: No module named 'xgboost'`

```bash
pip install xgboost
```

### Streamlit app can't find model files

Ensure you're launching from the project root:
```bash
cd LoanApprovalPrediction
streamlit run Streamlit_App/app.py
```

---

## 👤 Author

**AIML Summer Intern 2026**  
Motilal Nehru National Institute of Technology, Allahabad  

---

## 📄 License

MIT License — free to use for educational and research purposes.
