import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Diabetes AI | Health Insights",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f2f8ff 0%, #f5fffc 100%);
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #123b70, #087f8c);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 35px;
    margin-bottom: 8px;
}
.hero p {
    color: #e2f5ff;
    font-size: 16px;
}
.section-card {
    background: white;
    border: 1px solid #dce8f2;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
div[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #dce8f2;
}
.stButton > button, div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #087f8c, #1266a8);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    border: 1px solid #087f8c;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce8f2;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>AI-powered health indicator classification using machine learning</p>
    <p> Minor Project · Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Educational demonstration to diagnose diabetes "
    
)

# ---------- LOAD MODEL ----------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "diabetes_model_small.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_package(path):
    return joblib.load(path)

try:
    package = load_package(MODEL_PATH)
    model = package["model"]
    features = package["features"]
    threshold = package.get("threshold", 0.30)
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

# ---------- FIELD LABELS ----------
yes_no_labels = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check in the past 5 years",
    "Smoker": "Smoked in lifetime",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Heart disease or heart attack history",
    "PhysActivity": "Physical activity in the past 30 days",
    "Fruits": "Eat fruit at least once per day",
    "Veggies": "Eat vegetables at least once per day",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Have healthcare coverage",
    "NoDocbcCost": "Could not see a doctor due to cost",
    "DiffWalk": "Difficulty walking or climbing stairs"
}

# ---------- INPUT FORM ----------
st.markdown("## 📋 Health Indicator Assessment")
st.write("Enter the values below, then submit the form to see the model's output.")

with st.form("health_form"):
    st.markdown("### 🫀 Medical History")
    col1, col2 = st.columns(2)
    values = {}

    for i, feature in enumerate(features):
        if feature not in [
            "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "AnyHealthcare", "NoDocbcCost"
        ]:
            continue

        with (col1 if i % 2 == 0 else col2):
            values[feature] = st.selectbox(
                yes_no_labels[feature],
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                key="field_" + feature
            )

    st.markdown("---")
    st.markdown("### 🥗 Lifestyle & Mobility")
    col1, col2 = st.columns(2)

    lifestyle = [
        "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "DiffWalk"
    ]
    for i, feature in enumerate(lifestyle):
        if feature in features:
            with (col1 if i % 2 == 0 else col2):
                values[feature] = st.selectbox(
                    yes_no_labels[feature],
                    [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key="field_" + feature
                )

    st.markdown("---")
    st.markdown("### 📊 General Health")
    col1, col2 = st.columns(2)

    for feature in features:
        if feature == "BMI":
            with col1:
                values[feature] = st.number_input(
                    "BMI (body mass index)", min_value=10.0,
                    max_value=100.0, value=25.0, step=0.5
                )
        elif feature == "GenHlth":
            with col2:
                values[feature] = st.selectbox(
                    "General health",
                    [1, 2, 3, 4, 5],
                    format_func=lambda x: {
                        1: "1 - Excellent", 2: "2 - Very good",
                        3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"
                    }[x]
                )
        elif feature == "MentHlth":
            values[feature] = st.slider(
                "Days of poor mental health in the past 30 days",
                0, 30, 0
            )
        elif feature == "PhysHlth":
            values[feature] = st.slider(
                "Days of poor physical health in the past 30 days",
                0, 30, 0
            )

    st.markdown("---")
    st.markdown("### 👤 Personal & Socioeconomic Indicators")
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            if feature == "Sex":
                values[feature] = st.selectbox(
                    "Sex (dataset coding)", [0, 1],
                    format_func=lambda x: {
                        0: "Female (0)", 1: "Male (1)"
                    }[x]
                )
            elif feature == "Age":
                values[feature] = st.selectbox(
                    "Age group (coded category)", list(range(1, 101))
                )
            elif feature == "Education":
                values[feature] = st.selectbox(
                    "Education level (coded category)", list(range(1, 11))
                )
            elif feature == "Income":
                values[feature] = st.selectbox(
                    "Income level (coded category)", list(range(1, 11))
                )

    st.markdown("")
    submitted = st.form_submit_button("🔍 Predict Diabetes Class")

# ---------- PREDICTION RESULT ----------
if submitted:
    missing = [f for f in features if f not in values]

    if missing:
        st.error("Some model inputs are missing: " + ", ".join(missing))
    else:
        input_df = pd.DataFrame(
            [[values[f] for f in features]],
            columns=features
        )

        try:
            probability = float(model.predict_proba(input_df)[0, 1])
            prediction = int(probability >= threshold)

            st.markdown("---")
            st.markdown("## 📌 Model Prediction Result")

            result_col, probability_col = st.columns([1, 1])

            with result_col:
                if prediction == 1:
                    st.error("### Positive classification")
                    st.write(
                        "The model classified this input as class 1 "
                        "at the configured decision threshold."
                    )
                else:
                    st.success("### Negative classification")
                    st.write(
                        "The model classified this input as class 0 "
                        "at the configured decision threshold."
                    )

            with probability_col:
                st.metric(
                    "Model output for class 1",
                    f"{probability:.1%}"
                )
                st.progress(max(0.0, min(1.0, probability)))

            st.caption(
                f"Decision threshold used: {threshold:.2f}. "
                "The model output is not a calibrated personal risk estimate "
                "or a medical diagnosis."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748b;'>"
    "AI-Based Diabetes Prediction System "
    "</div>",
    unsafe_allow_html=True
)
