import streamlit as st
import joblib
import numpy as np

st.title("Titanic Survival Prediction App")

# Load trained model
model = joblib.load("titanic_model.joblib")

st.subheader("Enter Passenger Details")

# Passenger class (integer)
p_class = st.selectbox("Passenger Class (p_class)", [1, 2, 3])

# Sex mapping
sex = st.selectbox("Sex", ["male", "female"])
sex = 0 if sex == "male" else 1

# Age (INTEGER ONLY)
age = st.number_input("Age", min_value=0, max_value=100, value=25, step=1)

# Fare (INTEGER ONLY)
fare = st.number_input("Fare", min_value=0, max_value=600, value=32, step=1)

# Prepare features
features = np.array([[p_class, sex, age, fare]])

if st.button("Predict"):
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("Survived!")
    else:
        st.error("Did NOT Survive")