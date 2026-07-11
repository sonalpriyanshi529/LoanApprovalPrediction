"""
app.py — Loan Approval Prediction Streamlit Application
=========================================================
AIML Summer Internship 2026 | MNNIT Allahabad

Run:
    streamlit run Streamlit_App/app.py
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor | MNNIT",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Design tokens (from brand palette) ── */
:root {
    --bg:            #101415;
    --card:          #363A3B;
    --accent:        #10B981;
    --accent-soft:   rgba(16,185,129,0.14);
    --text-primary:  #FFFFFF;
    --text-secondary:#9CA3AF;
    --border:        #1F2937;
    --danger:        #F87171;
    --danger-soft:   rgba(248,113,113,0.14);
}

/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: var(--bg);
    color: var(--text-primary);
}

/* Tighten default top padding a touch for a denser dashboard feel */
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; }

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #14201C 0%, #182420 60%, #101415 100%);
    border: 1px solid var(--border);
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    color: var(--text-primary);
}
.main-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 0.35rem 0; letter-spacing: -0.02em; }
.main-header p  { font-size: 0.9rem; color: var(--text-secondary); margin: 0; }

/* ── Cards (metrics) ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent);
    box-shadow: 0 8px 22px rgba(16,185,129,0.18);
}
.metric-card .label { font-size: 0.72rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
.metric-card .value { font-size: 1.65rem; font-weight: 700; color: var(--accent); margin-top: 0.25rem; }
.metric-card .sub   { font-size: 0.72rem; color: var(--text-secondary); margin-top: 0.15rem; opacity: 0.85; }

/* ── Result boxes ── */
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.14), rgba(16,185,129,0.05));
    border: 1.5px solid var(--accent);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse 0.5s ease;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(248,113,113,0.14), rgba(248,113,113,0.05));
    border: 1.5px solid var(--danger);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse 0.5s ease;
}
@keyframes pulse {
    0%   { transform: scale(0.97); opacity: 0.7; }
    50%  { transform: scale(1.02); }
    100% { transform: scale(1.00); opacity: 1; }
}

/* ── Section titles ── */
.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    border-left: 3px solid var(--accent);
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem 0;
    letter-spacing: 0.01em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #14181A;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color: var(--text-secondary) !important; font-size: 0.82rem; }
[data-testid="stSidebar"] hr { border-color: var(--border); }

/* Sidebar info/metric boxes pick up card styling for consistency */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: var(--card);
    border: 1px solid var(--accent);
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.4rem;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 1.1rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Inputs ── */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
.stSelectbox div[data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
label, .stMarkdown p { color: var(--text-primary); }
[data-testid="stWidgetLabel"] p { color: var(--text-secondary) !important; font-size: 0.85rem; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #06231B !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.2rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    box-shadow: 0 4px 14px rgba(16,185,129,0.25);
}
.stButton > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(16,185,129,0.35);
}

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--text-primary);
}

/* ── Dataframe / tables ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow-x: auto; }

/* ── Divider ── */
hr { border-color: var(--border); margin: 1.5rem 0; }

/* ── Badge ── */
.badge {
    display: inline-block;
    background: var(--accent);
    color: #06231B;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 6px;
    vertical-align: middle;
}

/* ── Responsive tweaks (mobile / tablet) ── */
@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .main-header { padding: 1.4rem 1.4rem; border-radius: 12px; }
    .main-header h1 { font-size: 1.4rem; }
    .main-header p { font-size: 0.8rem; }
    .metric-card { padding: 0.9rem 0.8rem; }
    .metric-card .value { font-size: 1.3rem; }
    div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
    [data-testid="stDataFrame"] { overflow-x: auto; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all model artifacts with version-safe fallback."""
    base_paths = [
        Path("Model"),
        Path("../Model"),
        Path(__file__).parent.parent / "Model",
    ]
    model_dir = None
    for p in base_paths:
        if p.exists() and any(p.glob("*.pkl")):
            model_dir = p
            break

    if model_dir is None:
        return None, "❌ Model directory not found. Run `python train_model.py` first."

    try:
        artifacts = {
            "model"    : joblib.load(model_dir / "best_model.pkl"),
            "scaler"   : joblib.load(model_dir / "scaler.pkl"),
            "encoders" : joblib.load(model_dir / "label_encoders.pkl"),
            "features" : joblib.load(model_dir / "feature_names.pkl"),
            "metrics"  : joblib.load(model_dir / "model_metrics.pkl"),
        }
        return artifacts, None
    except Exception as e:
        msg = (
            f"❌ Failed to load model: {e}\n\n"
            "**Fix:** Python/sklearn version mismatch between Colab and local env.\n"
            "Run `python train_model.py` to regenerate compatible `.pkl` files."
        )
        return None, msg


def build_input_vector(inputs: dict, artifacts: dict) -> np.ndarray:
    """Transform raw UI inputs into model-ready feature vector."""
    le = artifacts["encoders"]
    fn = artifacts["features"]

    total = inputs["ApplicantIncome"] + inputs["CoapplicantIncome"]
    term  = max(inputs["Loan_Amount_Term"], 1)
    emi   = inputs["LoanAmount"] / term

    def safe_encode(col, val):
        enc = le[col]
        val_str = str(val)
        if val_str in enc.classes_:
            return enc.transform([val_str])[0]
        return 0  # fallback

    row = {
        "Gender"               : safe_encode("Gender",        inputs["Gender"]),
        "Married"              : safe_encode("Married",       inputs["Married"]),
        "Dependents"           : float(str(inputs["Dependents"]).replace("+","")),
        "Education"            : safe_encode("Education",     inputs["Education"]),
        "Self_Employed"        : safe_encode("Self_Employed", inputs["Self_Employed"]),
        "Loan_Amount_Term"     : float(inputs["Loan_Amount_Term"]),
        "Credit_History"       : float(inputs["Credit_History"]),
        "Property_Area"        : safe_encode("Property_Area", inputs["Property_Area"]),
        "Log_ApplicantIncome"  : np.log1p(inputs["ApplicantIncome"]),
        "Log_CoapplicantIncome": np.log1p(inputs["CoapplicantIncome"]),
        "Log_LoanAmount"       : np.log1p(inputs["LoanAmount"]),
        "Log_TotalIncome"      : np.log1p(total),
        "EMI"                  : emi,
        "Balance_Income"       : total - emi * 1000,
        "Debt_Income_Ratio"    : inputs["LoanAmount"] / max(total, 1),
    }

    X = pd.DataFrame([row])[fn]
    best = artifacts["metrics"]["best_model_name"]
    if best == "Logistic Regression":
        X = artifacts["scaler"].transform(X)
        return X
    return X.values


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Loan Approval Prediction System</h1>
        <p>AIML Summer Internship 2026 &nbsp;|&nbsp; Motilal Nehru National Institute of Technology, Allahabad</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Artifacts ──
    with st.spinner("Loading model..."):
        artifacts, err = load_artifacts()

    if err:
        st.error(err)
        st.stop()

    metrics     = artifacts["metrics"]
    best_name   = metrics["best_model_name"]
    all_results = metrics["results"]

    # ── SIDEBAR: Model Info ──
    with st.sidebar:
        st.markdown("## 🤖 Model Dashboard")
        st.markdown(f"**Active Model**")
        st.info(f"🏆 {best_name}")
        st.markdown("---")
        st.markdown("**📊 Performance Metrics**")
        bm = all_results[best_name]
        for k, v in bm.items():
            if k not in ("CV_Mean","CV_Std"):
                st.metric(k, f"{v*100:.1f}%")
        st.markdown("---")
        st.caption(f"Python: {sys.version.split()[0]}")
        st.caption(f"sklearn: {__import__('sklearn').__version__}")

    # ── TOP METRICS ROW ──
    st.markdown('<p class="section-title">📈 Model Performance Overview</p>', unsafe_allow_html=True)
    cols = st.columns(5)
    metric_labels = [
        ("Accuracy",  "Accuracy",  "Overall correct predictions"),
        ("Precision", "Precision", "Approved predictions correct"),
        ("Recall",    "Recall",    "Actual approved cases found"),
        ("F1_Score",  "F1-Score",  "Precision-Recall balance"),
        ("ROC_AUC",   "ROC-AUC",   "Discrimination ability"),
    ]
    for col, (key, label, sub) in zip(cols, metric_labels):
        val = bm[key]
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val*100:.1f}%</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── TABS ──
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Comparison", "ℹ️ About"])

    # ══════════════════════════════════════════
    # TAB 1 — PREDICT
    # ══════════════════════════════════════════
    with tab1:
        st.markdown('<p class="section-title">📋 Applicant Information</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Personal Details**")
            gender       = st.selectbox("Gender",       ["Male","Female"])
            married      = st.selectbox("Marital Status", ["Yes","No"])
            dependents   = st.selectbox("Dependents",   ["0","1","2","3+"])
            education    = st.selectbox("Education",    ["Graduate","Not Graduate"])
            self_employed= st.selectbox("Self Employed",["No","Yes"])

        with c2:
            st.markdown("**Financial Details**")
            applicant_income   = st.number_input("Applicant Income (₹/month)", 0, 1_000_000, 5000, 500)
            coapplicant_income = st.number_input("Co-Applicant Income (₹/month)", 0, 500_000, 0, 500)
            loan_amount        = st.number_input("Loan Amount (₹ thousands)", 10, 700, 150, 10)
            loan_term          = st.selectbox("Loan Term (months)", [360,180,480,120,240,60,300,36,84], index=0)
            property_area      = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

        with c3:
            st.markdown("**Credit Profile**")
            credit_history = st.radio("Credit History", [1.0, 0.0],
                                       format_func=lambda x: "✅ Meets Guidelines" if x==1.0 else "❌ Does Not Meet")
            st.markdown("")
            st.info(f"""
            **Quick Summary**
            - Total Monthly Income: ₹{applicant_income + coapplicant_income:,}
            - EMI Estimate: ₹{loan_amount*1000/max(loan_term,1):,.0f}/month
            - Debt-Income Ratio: {loan_amount/(max((applicant_income+coapplicant_income)/1000,0.01)):.2f}x
            """)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Loan Approval", use_container_width=True, type="primary")

        if predict_btn:
            inputs = {
                "Gender"            : gender,
                "Married"           : married,
                "Dependents"        : dependents,
                "Education"         : education,
                "Self_Employed"     : self_employed,
                "ApplicantIncome"   : applicant_income,
                "CoapplicantIncome" : coapplicant_income,
                "LoanAmount"        : loan_amount,
                "Loan_Amount_Term"  : loan_term,
                "Credit_History"    : credit_history,
                "Property_Area"     : property_area,
            }

            with st.spinner("Analysing application..."):
                X_input  = build_input_vector(inputs, artifacts)
                model    = artifacts["model"]
                pred     = model.predict(X_input)[0]
                prob     = model.predict_proba(X_input)[0]
                approved = int(pred) == 1
                conf     = prob[1] if approved else prob[0]

                if approved:
                    st.markdown(f"""
                    <div class="result-approved">
                        <h2 style="color:#10B981; margin:0;">✅ LOAN APPROVED</h2>
                        <p style="font-size:1.1rem; color:#FFFFFF; margin:0.5rem 0">
                            Confidence: <strong>{conf*100:.1f}%</strong>
                        </p>
                        <p style="color:#9CA3AF; font-size:0.9rem;">
                            The applicant's profile meets our lending criteria.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                <div class="result-rejected">
                    <h2 style="color:#F87171; margin:0;">❌ LOAN REJECTED</h2>
                    <p style="font-size:1.1rem; color:#FFFFFF; margin:0.5rem 0">
                        Confidence: <strong>{conf*100:.1f}%</strong>
                    </p>
                    <p style="color:#9CA3AF; font-size:0.9rem;">
                        The application does not meet current lending criteria.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            # Probability breakdown
            cc1, cc2 = st.columns(2)
            cc1.metric("Approval Probability",  f"{prob[1]*100:.1f}%",
                       delta=f"{(prob[1]-0.5)*100:+.1f}% vs baseline")
            cc2.metric("Rejection Probability", f"{prob[0]*100:.1f}%")

            # Key risk factors
            st.markdown('<p class="section-title">💡 Key Factors</p>', unsafe_allow_html=True)
            factors = {
                "Credit History"  : ("✅ Good" if credit_history==1.0 else "⚠️ Poor", credit_history==1.0),
                "Total Income"    : (f"₹{applicant_income+coapplicant_income:,}/mo",
                                    (applicant_income+coapplicant_income) > 5000),
                "Loan Amount"     : (f"₹{loan_amount}K",
                                    loan_amount < 200),
                "Education"       : ("✅ Graduate" if education=="Graduate" else "ℹ️ Non-Graduate",
                                    education=="Graduate"),
                "EMI Burden"      : (f"₹{loan_amount*1000/max(loan_term,1):,.0f}/mo",
                                    (loan_amount*1000/max(loan_term,1)) < applicant_income * 0.4),
            }
            f_cols = st.columns(len(factors))
            for col, (fname, (fval, fpos)) in zip(f_cols, factors.items()):
                color = "#10B981" if fpos else "#F87171"
                col.markdown(f"""
                <div style="background:{color}18;border:1px solid {color}40;
                            border-radius:10px;padding:0.6rem 0.8rem;text-align:center;">
                    <div style="font-size:0.7rem;color:{color};font-weight:600">{fname}</div>
                    <div style="font-size:0.85rem;font-weight:600;margin-top:0.2rem;color:#FFFFFF">{fval}</div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TAB 2 — MODEL COMPARISON
    # ══════════════════════════════════════════
    with tab2:
        st.markdown('<p class="section-title">📊 All Models Performance Comparison</p>', unsafe_allow_html=True)

        rows = []
        for mname, mmetrics in all_results.items():
            row = {"Model": mname}
            row.update({k: f"{v*100:.2f}%" for k, v in mmetrics.items()
                        if k not in ("CV_Std",)})
            row["Best"] = "🏆" if mname == best_name else ""
            rows.append(row)

        cdf = pd.DataFrame(rows).set_index("Model")
        st.dataframe(cdf, use_container_width=True, height=180)

        st.markdown('<p class="section-title">📈 Metric Breakdown</p>', unsafe_allow_html=True)
        metric_keys = ["Accuracy","Precision","Recall","F1_Score","ROC_AUC"]
        chart_data  = pd.DataFrame(
            {m: [all_results[mn][m] for mn in all_results] for m in metric_keys},
            index=list(all_results.keys())
        )
        st.bar_chart(chart_data, use_container_width=True, height=350)

        st.markdown('<p class="section-title">📋 Feature Engineering Applied</p>', unsafe_allow_html=True)
        fe_info = {
            "TotalIncome"       : "Applicant + Co-applicant income",
            "Log_*"             : "Log-transform to reduce skewness",
            "EMI"               : "LoanAmount / Loan_Amount_Term",
            "Balance_Income"    : "TotalIncome − (EMI × 1000)",
            "Debt_Income_Ratio" : "LoanAmount / TotalIncome",
        }
        for feat, desc in fe_info.items():
            st.markdown(f"- **`{feat}`** — {desc}")

    # ══════════════════════════════════════════
    # TAB 3 — ABOUT
    # ══════════════════════════════════════════
    with tab3:
        st.markdown("""
        ## 🏦 Loan Approval Prediction System

        **Project:** AIML Summer Internship 2026  
        **Institute:** Motilal Nehru National Institute of Technology, Allahabad  

        ---

        ### 🎯 Objective
        Automate the initial screening of loan applications using Machine Learning,
        reducing processing time and improving consistency.

        ### 📋 ML Lifecycle
        1. **Problem Understanding** — Binary classification (Approved/Rejected)
        2. **Data Collection** — Kaggle Loan Prediction Dataset (614 records)
        3. **Data Preprocessing** — Imputation, encoding, scaling
        4. **Exploratory Data Analysis** — Distribution, correlation, class balance
        5. **Feature Engineering** — EMI, Balance Income, Log transforms, Debt Ratio
        6. **Model Building** — Logistic Regression, Random Forest, XGBoost
        7. **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC + 5-fold CV
        8. **Deployment** — This Streamlit application

        ### 🗂️ Project Structure
        ```
        LoanApprovalPrediction/
        ├── Dataset/              ← loan_data.csv
        ├── Notebook/             ← Colab notebook + EDA plots
        ├── Model/                ← best_model.pkl, model_metrics.pkl, scaler.pkl, ...
        ├── Streamlit_App/        ← app.py (this file)
        ├── Documentation/        ← Project report
        ├── train_model.py        ← Local retraining script
        └── README.md
        ```

        ### 📦 Technologies
        | Tool | Purpose |
        |------|---------|
        | Python 3.10+ | Core language |
        | scikit-learn | ML models & preprocessing |
        | XGBoost | Gradient boosting model |
        | pandas / numpy | Data manipulation |
        | Streamlit | Web deployment |
        | joblib | Model serialisation |
        | Google Colab | Development environment |

        ### ⚡ Version Mismatch Fix
        If the Streamlit app throws a loading error due to Python/sklearn version
        differences between Colab and your local machine:

        ```bash
        pip install -r requirements.txt
        python train_model.py      # regenerates compatible .pkl files
        streamlit run Streamlit_App/app.py
        ```
        """)


if __name__ == "__main__":
    main()
