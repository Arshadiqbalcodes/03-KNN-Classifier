import pickle
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Diabetes AI Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071A1F, #0B2A2F, #071A1F);
    color: white;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 25px;
    margin-bottom: 25px;
}

.hero-icon {
    font-size: 55px;
}

.hero h1 {
    font-size: 45px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero p {
    color: #9FB8B8;
    font-size: 17px;
}

.section {
    background: rgba(17, 43, 48, 0.85);
    border: 1px solid rgba(89, 180, 180, 0.18);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.20);
}

.section-title {
    font-size: 23px;
    font-weight: 600;
    margin-bottom: 20px;
}

.stNumberInput label,
.stSlider label {
    color: #B8CECE !important;
    font-weight: 500 !important;
}

.stNumberInput input {
    background-color: #0A2025 !important;
    color: white !important;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #20C7B5, #13A99E);
    color: white;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #27DCC8, #18B7AA);
    color: white;
}

.result-positive {
    background: rgba(255, 78, 78, 0.12);
    border: 1px solid #FF5C5C;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
}

.result-negative {
    background: rgba(32, 199, 181, 0.12);
    border: 1px solid #20C7B5;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
}

.result-icon {
    font-size: 55px;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
}

.confidence {
    font-size: 18px;
    color: #B8CECE;
    margin-top: 10px;
}

.info-box {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 15px;
    margin-top: 20px;
    color: #AFC4C4;
    font-size: 14px;
    text-align: center;
}

.footer {
    text-align: center;
    color: #718989;
    margin-top: 30px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("knn_diabetes_model.pkl", "rb") as file:
            return pickle.load(file)
    except Exception as error:
        st.error(f"Model loading error: {error}")
        return None

model = load_model()

st.markdown("""
<div class="hero">
<div class="hero-icon">🩺</div>
<h1>Diabetes AI Classifier</h1>
<p>K-Nearest Neighbors Machine Learning Model</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
<div class="section-title">👤 Patient Information</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    glucose = st.slider(
        "Glucose (mg/dL)",
        min_value=0,
        max_value=250,
        value=120
    )

    blood_pressure = st.slider(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=140,
        value=70
    )

    skin_thickness = st.slider(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=25
    )

with col2:
    insulin = st.slider(
        "Insulin (μU/mL)",
        min_value=0,
        max_value=850,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=30.5,
        step=0.1,
        format="%.1f"
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.40,
        step=0.01,
        format="%.2f"
    )

    age = st.slider(
        "Age",
        min_value=1,
        max_value=100,
        value=35
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

_, button_col, _ = st.columns([1, 2, 1])

with button_col:
    predict = st.button(
        "🔍 ANALYZE PATIENT",
        use_container_width=True
    )

if predict:
    if model is None:
        st.error(
            "Model could not be loaded. "
            "Make sure knn_diabetes_model.pkl is in the same folder as app.py."
        )
    else:
        try:
            sample = np.array([[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]])

            prediction = model.predict(sample)[0]
            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                classes = list(model.classes_)

                if prediction in classes:
                    index = classes.index(prediction)
                    confidence = probabilities[index] * 100

            confidence_text = (
                f"{confidence:.1f}%"
                if confidence is not None
                else "N/A"
            )

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-positive">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">Diabetes Detected</div>
                    <div class="confidence">
                    Model Confidence: <b>{confidence_text}</b>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-negative">
                    <div class="result-icon">✅</div>
                    <div class="result-title">No Diabetes Detected</div>
                    <div class="confidence">
                    Model Confidence: <b>{confidence_text}</b>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("""
            <div class="section">
            <div class="section-title">📊 Patient Data Summary</div>
            </div>
            """, unsafe_allow_html=True)

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            summary_col1.metric("Glucose", f"{glucose} mg/dL")
            summary_col2.metric("BMI", f"{bmi:.1f}")
            summary_col3.metric("Blood Pressure", f"{blood_pressure}")
            summary_col4.metric("Age", f"{age}")

        except Exception as error:
            st.error(f"Prediction Error: {error}")

st.markdown("""
<div class="info-box">
⚕️ <b>Disclaimer:</b>
This application is created for educational and machine-learning purposes.
It is not a medical diagnostic tool. Please consult a qualified healthcare
professional for medical advice.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
KNN Diabetes Classifier • Built with Python, Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)