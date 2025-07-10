import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load Random Forest model and preprocessors
model = joblib.load("side_effect_rf_model.pkl")
preprocessor = joblib.load("preprocessor_rf.pkl")
mlb_top50 = joblib.load("mlb_top50.pkl")

# Streamlit UI
st.title("💊 Drug Side Effect Predictor (Random Forest)")
st.write("Enter drug details to predict possible side effects.")

# User Inputs
drug_name = st.text_input("Drug Name", "Adderall")
medical_condition = st.text_input("Medical Condition", "ADHD")
generic_name = st.text_input("Generic Name", "amphetamine / dextroamphetamine")
drug_classes = st.text_input("Drug Class", "CNS stimulants")
rx_otc = st.selectbox("Rx/OTC", ["Rx", "OTC", "Unknown"])
pregnancy_category = st.selectbox("Pregnancy Category", ["A", "B", "C", "D", "X", "N", "Unknown"])
csa = st.text_input("CSA Schedule", "Schedule 2")
rating = st.slider("User Rating", 1.0, 10.0, 8.2)
no_of_reviews = st.number_input("Number of Reviews", 0, 5000, 248)

# Predict button
if st.button("Predict Side Effects"):
    # Prepare input
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

    # Transform and predict
    X_new = preprocessor.transform(new_input)
    y_pred = model.predict(X_new)[0]  # binary prediction

    import numpy as np
    predicted_effects = mlb_top50.inverse_transform(np.array([y_pred]))[0]


    if predicted_effects:
        st.success("Top Predicted Side Effects:")
        for effect in predicted_effects:
            st.write(f"• **{effect}**")
        
        # Optional: Show bar chart of selected effects
        st.subheader("Predicted Effects Chart")
        fig, ax = plt.subplots()
        ax.barh(predicted_effects[::-1], [1]*len(predicted_effects), color='lightcoral')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted")
        ax.set_title("Side Effects")
        st.pyplot(fig)
    else:
        st.info("No side effects predicted confidently.")
