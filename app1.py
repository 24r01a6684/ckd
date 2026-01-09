import streamlit as st
import pickle
import numpy as np

# Load the fine-tuned models
with open("RandomForest_model.pkl", "rb") as file:
    model_tuned = pickle.load(file)
with open("AdaBoost_model.pkl", "rb") as file:
    model_adb = pickle.load(file)
with open("GradientBoosting_model.pkl", "rb") as file:
    model_rf = pickle.load(file)

# Define input fields for user
st.title("Kidney Disease Prediction")
st.write("Enter the values below to predict whether a person has CKD or not.")

# Feature inputs with descriptions
sg = st.number_input("Specific Gravity (sg)", min_value=1.005, max_value=1.05, step=0.001)
with st.expander("Specific Gravity (sg) "):
    st.write(
        """It measures how concentrated the urine is.  
        **Normal range:** 1.005 - 1.030  
        **If low:** Too much water in urine (could be kidney issues).  
        **If high:** Less water, more substances (dehydration or kidney problems)."""
    )
htn = st.selectbox("Hypertension (htn)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with st.expander("Hypertension (htn)"):
    st.write(
    """Hypertension = High blood pressure (BP).
    If present, it means blood pressure is higher than normal, which can affect kidney health.
    High BP is a risk factor for kidney disease.
    """
    )

hemo = st.number_input("Hemoglobin (hemo) in gms", min_value=3.0, max_value=17.0, step=0.1)
with st.expander("Hemoglobin (hemo)"):
    st.write(
    """
Hemoglobin is a protein in red blood cells that carries oxygen.

*Normal range:* 

**Men**: 13.8–17.2 g/dL

**Women**: 12.1–15.1 g/dL

Low levels → Anemia (tiredness, weakness).

High levels → Dehydration or other conditions.



    """
    )

dm = st.selectbox("Diabetes Mellitus (dm)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with st.expander("Diabetes Mellitus (dm)"):
    st.write(
    """
    Diabetes Mellitus = High blood sugar levels.

    If present, it means the person has diabetes, which can damage kidneys over time.
    """
    )

al = st.slider("Albumin (al)", min_value=0, max_value=5, step=1)
with st.expander("Albumin (al)"):
    st.write(
    """
    Albumin is a protein in blood.
    
    **Normal range**: 0-5 mg/dL.
    
    High levels in urine → Kidney damage or disease.
    """
    )

appet = st.selectbox("Appetite (appet)", [0, 1], format_func=lambda x: "Good" if x == 1 else "Poor")
with st.expander("Appetite (appet)"):
    st.write(
    """
    Appetite = Desire to eat.

    Poor appetite can indicate health issues, including kidney problems.
    """
    )

rc = st.number_input("Red Blood Cell Count (rc) in millions/cmm", min_value=2.0, max_value=7.0, step=0.1)
with st.expander("Red Blood Cell Count (rc)"):
    st.write(
    """
    Red Blood Cell Count (rc) - 4.50 million/cmm

Measures the number of red blood cells in the blood.

*Normal range*:

**Men**: 4.7–6.1 million/cmm

**Women**: 4.2–5.4 million/cmm

**Low RBC** → Anemia (fatigue, weakness).

**High RBC** → Can indicate dehydration or other issues.

    """
    )

pc = st.selectbox("Pus Cell (pc)", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
with st.expander("Pus Cell (pc)"):
    st.write(
    """
    Pus Cell = White blood cells in urine.

    **Normal**: 0-5 pus cells per high power field (HPF).

    High levels → Infection or inflammation in urinary tract or kidneys.
    """
    )

# Prepare data for prediction
data = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])

# Model selection buttons
st.write("Select a model to make a prediction:")
if st.button("Random Forest Model"):
    prediction = model_rf.predict(data)[0]
    accuracy = "97.5%"
    selected_model = "Random Forest"
elif st.button("AdaBoost Model"):
    prediction = model_adb.predict(data)[0]
    accuracy = "96.6%"
    selected_model = "AdaBoost"
elif st.button("Gradient Boosting Model"):
    prediction = model_tuned.predict(data)[0]
    accuracy = "97.5%"
    selected_model = "Gradient Boosting "
else:
    prediction = None
    accuracy = None
    selected_model = None

# Display results
if prediction is not None:
    st.write(f"**Selected Model:** {selected_model} (Accuracy: {accuracy})")
    if prediction == 1:
        st.error("Prediction: The person is likely to have Chronic Kidney Disease (CKD).")
    else:
        st.success("Prediction: The person is not likely to have CKD.")