import streamlit as st
import joblib
import numpy as np

st.title("Titanic Survival Prediction App")

# Load trained model
model = joblib.load("titanic_model.joblib")

st.subheader("Enter Passenger Details")

p_class = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["male", "female"])
sex = 0 if sex == "male" else 1

age = st.number_input("Age", 0, 100, 25, step=1)
fare = st.number_input("Fare", 0, 600, 32, step=1)

features = np.array([[p_class, sex, age, fare]])

if st.button("Predict"):
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("Survived!")
    else:
        st.error("Did NOT Survive")