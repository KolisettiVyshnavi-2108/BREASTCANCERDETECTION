import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------
# Load model and scaler
# ---------------------------------
model = tf.keras.models.load_model("BC_model.h5")

with open("BC_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------
# Feature name mapping (UI → Training)
# ---------------------------------
feature_mapping = {
    "radius_mean": "mean radius",
    "texture_mean": "mean texture",
    "perimeter_mean": "mean perimeter",
    "area_mean": "mean area",
    "smoothness_mean": "mean smoothness",
    "compactness_mean": "mean compactness",
    "concavity_mean": "mean concavity",
    "concave points_mean": "mean concave points",
    "symmetry_mean": "mean symmetry",
    "fractal_dimension_mean": "mean fractal dimension",

    "radius_se": "radius error",
    "texture_se": "texture error",
    "perimeter_se": "perimeter error",
    "area_se": "area error",
    "smoothness_se": "smoothness error",
    "compactness_se": "compactness error",
    "concavity_se": "concavity error",
    "concave points_se": "concave points error",
    "symmetry_se": "symmetry error",
    "fractal_dimension_se": "fractal dimension error",

    "radius_worst": "worst radius",
    "texture_worst": "worst texture",
    "perimeter_worst": "worst perimeter",
    "area_worst": "worst area",
    "smoothness_worst": "worst smoothness",
    "compactness_worst": "worst compactness",
    "concavity_worst": "worst concavity",
    "concave points_worst": "worst concave points",
    "symmetry_worst": "worst symmetry",
    "fractal_dimension_worst": "worst fractal dimension"
}

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("Breast Cancer Prediction")
st.write("Enter the tumor feature values and predict the class.")

st.subheader("Input Features")

input_data = {
    "radius_mean": st.number_input("radius_mean", value=14.0),
    "texture_mean": st.number_input("texture_mean", value=20.0),
    "perimeter_mean": st.number_input("perimeter_mean", value=90.0),
    "area_mean": st.number_input("area_mean", value=600.0),
    "smoothness_mean": st.number_input("smoothness_mean", value=0.10),
    "compactness_mean": st.number_input("compactness_mean", value=0.15),
    "concavity_mean": st.number_input("concavity_mean", value=0.20),
    "concave points_mean": st.number_input("concave points_mean", value=0.10),
    "symmetry_mean": st.number_input("symmetry_mean", value=0.20),
    "fractal_dimension_mean": st.number_input("fractal_dimension_mean", value=0.06),

    "radius_se": st.number_input("radius_se", value=0.20),
    "texture_se": st.number_input("texture_se", value=1.00),
    "perimeter_se": st.number_input("perimeter_se", value=1.50),
    "area_se": st.number_input("area_se", value=20.00),
    "smoothness_se": st.number_input("smoothness_se", value=0.01),
    "compactness_se": st.number_input("compactness_se", value=0.02),
    "concavity_se": st.number_input("concavity_se", value=0.03),
    "concave points_se": st.number_input("concave points_se", value=0.01),
    "symmetry_se": st.number_input("symmetry_se", value=0.03),
    "fractal_dimension_se": st.number_input("fractal_dimension_se", value=0.004),

    "radius_worst": st.number_input("radius_worst", value=16.00),
    "texture_worst": st.number_input("texture_worst", value=25.00),
    "perimeter_worst": st.number_input("perimeter_worst", value=105.00),
    "area_worst": st.number_input("area_worst", value=800.00),
    "smoothness_worst": st.number_input("smoothness_worst", value=0.12),
    "compactness_worst": st.number_input("compactness_worst", value=0.20),
    "concavity_worst": st.number_input("concavity_worst", value=0.30),
    "concave points_worst": st.number_input("concave points_worst", value=0.15),
    "symmetry_worst": st.number_input("symmetry_worst", value=0.25),
    "fractal_dimension_worst": st.number_input("fractal_dimension_worst", value=0.08),
}

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Rename columns to training feature names
    input_df.rename(columns=feature_mapping, inplace=True)

    # Reorder columns exactly as scaler expects
    input_df = input_df[scaler.feature_names_in_]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0][0]

    result = "Malignant" if prediction > 0.5 else "Benign"

    st.success(f"Prediction: **{result}**")
    st.info(f"Prediction Probability: **{prediction:.4f}**")
