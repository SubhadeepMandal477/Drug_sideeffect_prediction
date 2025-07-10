import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Load model and helpers
model = load_model("side_effect_predictor.keras")
preprocessor = joblib.load("preprocessor.pkl")
mlb_top50 = joblib.load("mlb_top50.pkl")

# UI
st.title("💊 Drug Side Effect Predictor")
st.write("Enter drug details to predict possible side effects.")

# Input fields
drug_name = st.text_input("Drug Name", "Adderall")
medical_condition = st.text_input("Medical Condition", "ADHD")
generic_name = st.text_input("Generic Name", "amphetamine / dextroamphetamine")
drug_classes = st.text_input("Drug Class", "CNS stimulants")
rx_otc = st.selectbox("Rx/OTC", ["Rx", "OTC", "Unknown"])
pregnancy_category = st.selectbox("Pregnancy Category", ["A", "B", "C", "D", "X", "N", "Unknown"])
csa = st.text_input("CSA Schedule", "Schedule 2")
rating = st.slider("User Rating", 1.0, 10.0, 8.2)
no_of_reviews = st.number_input("Number of Reviews", 0, 5000, 248)

# ✅ All prediction logic stays inside this block!
if st.button("Predict Side Effects"):

    # Create input DataFrame
    new_input = pd.DataFrame([{
        "drug_name": drug_name,
        "medical_condition": medical_condition,
        "generic_name": generic_name,
        "drug_classes": drug_classes,
        "rx_otc": rx_otc,
        "pregnancy_category": pregnancy_category,
        "csa": csa,
        "rating": rating,
        "no_of_reviews": no_of_reviews
    }])

    # Preprocess
    X_new = preprocessor.transform(new_input)
    y_raw = model.predict(X_new.toarray())[0]

    # Top 5 predictions
    top_n = 10
    top_indices = np.argsort(y_raw)[-top_n:][::-1]
    top_effects = [(mlb_top50.classes_[i], y_raw[i]) for i in top_indices]

    # Show predictions
    st.success("Top Predicted Side Effects with Confidence:")
    for effect, score in top_effects:
        st.write(f"• **{effect}** — {score:.2f}")

    # Plot chart
    st.subheader("Confidence Chart")
    effects = [e[0] for e in top_effects]
    scores = [e[1] for e in top_effects]

    fig, ax = plt.subplots()
    ax.barh(effects[::-1], scores[::-1], color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Top 5 Side Effects")
    st.pyplot(fig)
