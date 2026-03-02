import streamlit as st
import pandas as pd
import joblib

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return joblib.load("titanic_svm_pipeline.pkl")

model = load_model()

# ==============================
# UI Header
# ==============================

st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
This app predicts whether a passenger would survive the Titanic disaster.
Fill in the passenger details below and click **Predict**.
""")

st.divider()

# ==============================
# User Input Section
# ==============================

st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    fare = st.number_input("Fare", min_value=0.0, value=32.0)

with col2:
    sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, value=0)
    parch = st.number_input("Number of Parents/Children", min_value=0, value=0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    alone = st.selectbox("Traveling Alone?", [True, False])

st.divider()

# ==============================
# Prediction Button
# ==============================

if st.button("Predict Survival"):

    # Create dataframe
    input_data = pd.DataFrame({
        "pclass": [pclass],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "sex": [sex],
        "embarked": [embarked],
        "alone": [alone]
    })

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ The passenger is likely to SURVIVE.")
    else:
        st.error("❌ The passenger is NOT likely to survive.")

    st.write(f"Survival Probability: **{round(probability * 100, 2)}%**")

    st.progress(float(probability))

st.divider()

# ==============================
# Footer
# ==============================

st.markdown("""
---
Developed using Machine Learning and Streamlit  
Model: Tuned SVM Pipeline  
""")