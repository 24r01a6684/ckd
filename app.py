import streamlit as st
import pickle
import numpy as np

# Load the models
with open("RandomForest_model.pkl", "rb") as file:
    rf_model = pickle.load(file)
with open("AdaBoost_model.pkl", "rb") as file:
    adb_model = pickle.load(file)
with open("GradientBoosting_model.pkl", "rb") as file:
    gb_model = pickle.load(file)

# Model accuracies (from training results)
model_accuracies = {
    "Random Forest": 0.78,  # Replace with actual accuracy
    "AdaBoost": 0.75,  # Replace with actual accuracy
    "Gradient Boosting": 0.77  # Replace with actual accuracy
}

st.title("Kidney Disease Prediction")
st.write("Select a model and enter the values below to predict CKD.")

# Model selection
selected_model = st.radio("Choose a Model", list(model_accuracies.keys()))

# Feature inputs
sg = st.number_input("Specific Gravity (sg)", min_value=1.005, max_value=1.025, step=0.001)
htn = st.selectbox("Hypertension (htn)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
hemo = st.number_input("Hemoglobin (hemo) in gms", min_value=3.0, max_value=17.0, step=0.1)
dm = st.selectbox("Diabetes Mellitus (dm)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
al = st.slider("Albumin (al)", min_value=0, max_value=5, step=1)
appet = st.selectbox("Appetite (appet)", [0, 1], format_func=lambda x: "Good" if x == 1 else "Poor")
rc = st.number_input("Red Blood Cell Count (rc) in millions/cmm", min_value=2.0, max_value=7.0, step=0.1)
pc = st.selectbox("Pus Cell (pc)", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")

data = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])

if st.button("Predict"):
    model = rf_model if selected_model == "Random Forest" else adb_model if selected_model == "AdaBoost" else gb_model
    prediction = model.predict(data)[0]
    accuracy = model_accuracies[selected_model]
    
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    if prediction == 1:
        st.error("Prediction: The person is likely to have Chronic Kidney Disease (CKD).")
    else:
        st.success("Prediction: The person is not likely to have CKD.")
