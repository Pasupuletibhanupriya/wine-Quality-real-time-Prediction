import streamlit as st
import numpy as np
import pickle

# =======================
# Load model and scaler
# =======================
with open("finalized_model.sav", "rb") as f:
    model = pickle.load(f)

with open("scaler_model.sav", "rb") as f:
    scaler = pickle.load(f)

# =======================
# Page config
# =======================
st.set_page_config(
    page_title="Wine Quality Prediction",
    layout="centered"
)

# =======================
# Background Image (CSS only for background)
# =======================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images5.alphacoders.com/443/thumb-1920-443997.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =======================
# Title
# =======================
st.title("🍷 Wine Quality Prediction App")
st.write("Enter wine chemical properties to predict quality")

# =======================
# Inputs
# =======================
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.01)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.01)
ph = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# =======================
# Predict Button + Result
# =======================
if st.button("🍷 Predict Wine Quality"):

    input_data = np.array([[ 
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]])

    scaled_data = scaler.transform(input_data)
    prediction = int(model.predict(scaled_data)[0])

    # Category logic
    if prediction <= 4:
        category = "❌ Bad Quality"
    elif prediction <= 6:
        category = "⚠️ Average Quality"
    elif prediction <= 8:
        category = "✅ Good Quality"
    else:
        category = "🌟 Excellent Quality"

    # Streamlit green success message (NO CSS)
    st.success("Prediction completed successfully ✔️")

    # Output (simple text as requested)
    st.write(f"Wine Quality Score: {prediction}")
    st.write(f"Category: {category}")

