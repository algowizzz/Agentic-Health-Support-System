import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="MediRisk AI", layout="wide")

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False
if "risk_prob" not in st.session_state:
    st.session_state.risk_prob = 0.0
if "feature_imp" not in st.session_state:
    st.session_state.feature_imp = {}

# ---------------------------------------------------
# SAFE MODEL LOADING (WORKS LOCAL + CLOUD)
# ---------------------------------------------------
@st.cache_resource
def load_models():
    try:
        # Absolute safe path
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

        models = {
            "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
            "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
            "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        }

        return models

    except Exception as e:
        st.error("🚨 Model loading failed")
        st.error(str(e))
        return {}

all_models = load_models()

# ---------------------------------------------------
# INPUT PREPROCESSING
# ---------------------------------------------------
def preprocess_inputs(age, gender, cp, trestbps, chol,
                      glucose, restecg, thalach,
                      exang, oldpeak, slope, ca, thal):

    sex = 1 if gender == "Male" else 0
    fbs = 1 if glucose > 120 else 0
    exang = 1 if exang == "Yes" else 0

    cp_map = {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-anginal Pain": 3,
        "Asymptomatic": 4
    }

    restecg_map = {
        "Normal": 0,
        "ST-T abnormality": 1,
        "LV hypertrophy": 2
    }

    slope_map = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    }

    thal_map = {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect": 7
    }

    processed = [
        float(age),
        sex,
        cp_map.get(cp, 4),
        float(trestbps),
        float(chol),
        fbs,
        restecg_map.get(restecg, 0),
        float(thalach),
        exang,
        float(oldpeak),
        slope_map.get(slope, 2),
        float(ca),
        thal_map.get(thal, 3)
    ]

    return pd.DataFrame([processed], columns=FEATURE_COLUMNS)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
def predict_risk(model, input_df):
    return float(model.predict_proba(input_df)[0][1])

def extract_feature_importance(model):
    inner = model
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model", model)

    if hasattr(inner, "feature_importances_"):
        importance = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        importance = np.abs(inner.coef_[0])
        importance = importance / np.sum(importance)
    else:
        importance = np.zeros(len(FEATURE_COLUMNS))

    feature_dict = dict(zip(FEATURE_COLUMNS, importance))
    return dict(sorted(feature_dict.items(), key=lambda x: x[1], reverse=True))

def categorize_risk(prob):
    if prob < 0.4:
        return "LOW"
    elif prob < 0.7:
        return "MODERATE"
    else:
        return "HIGH"

# ---------------------------------------------------
# UI STYLING
# ---------------------------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #0b1220;
    color: white;
}

.risk-wrapper {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}

.risk-circle {
    width: 190px;
    height: 190px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.6s ease-in-out;
}

.risk-inner {
    width: 150px;
    height: 150px;
    background: #0b1220;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.risk-percent {
    font-size: 2.4rem;
    font-weight: bold;
}

.risk-label {
    font-size: 1.2rem;
    font-weight: 600;
    text-align: center;
    margin-top: 12px;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0px rgba(239,68,68,0.5); }
    50% { box-shadow: 0 0 25px rgba(239,68,68,0.9); }
    100% { box-shadow: 0 0 0px rgba(239,68,68,0.5); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("Patient Risk Assessment System")
st.caption("AI-based clinical risk prediction using Cleveland Heart Disease dataset")
st.divider()

left, right = st.columns([1.5, 1])

with left:
    st.subheader("Patient Clinical Information")

    age = st.number_input("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 190)
    thalach = st.slider("Maximum Heart Rate", 40, 200, 150)
    glucose = st.slider("Fasting Blood Sugar", 50, 300, 100)
    restecg = st.selectbox("Resting ECG", [
        "Normal",
        "ST-T abnormality",
        "LV hypertrophy"
    ])
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Major Vessels (0-4)", 0, 4, 0)
    thal = st.selectbox("Thalassemia", [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ])
    model_name = st.selectbox("Model Selection", list(all_models.keys()))

with right:
    st.subheader("Assessment Result")

    if st.button("Run Risk Assessment", use_container_width=True):

        if model_name not in all_models:
            st.error("Model not loaded properly.")
            st.stop()

        with st.spinner("Running ML inference..."):
            time.sleep(1)

            try:
                input_df = preprocess_inputs(
                    age, gender, cp, trestbps, chol,
                    glucose, restecg, thalach,
                    exang, oldpeak, slope, ca, thal
                )

                model = all_models[model_name]
                prob = predict_risk(model, input_df)
                importance = extract_feature_importance(model)

            except Exception as e:
                st.error("Prediction failed.")
                st.error(str(e))
                st.stop()

            st.session_state.risk_prob = prob
            st.session_state.feature_imp = dict(list(importance.items())[:5])
            st.session_state.analysis_run = True

    if st.session_state.analysis_run:
        prob = st.session_state.risk_prob
        level = categorize_risk(prob)
        percent = int(prob * 100)

        if level == "LOW":
            color = "#22c55e"
        elif level == "MODERATE":
            color = "#facc15"
        else:
            color = "#ef4444"

        pulse = "animation: pulse 1.5s infinite;" if level == "HIGH" else ""

        st.markdown(f"""
        <div class="risk-wrapper">
            <div class="risk-circle" style="background: conic-gradient({color} {percent}%, #1f2937 {percent}%); {pulse}">
                <div class="risk-inner">
                    <div class="risk-percent" style="color:{color};">{percent}%</div>
                </div>
            </div>
        </div>
        <div class="risk-label" style="color:{color};">{level} RISK</div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Top Contributing Factors")
        for feature, value in st.session_state.feature_imp.items():
            st.write(f"{feature}: {round(value*100,2)}%")

st.divider()
st.caption("MediRisk AI • Clinical Demo Version")