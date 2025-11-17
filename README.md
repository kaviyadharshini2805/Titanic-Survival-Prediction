# Titanic Survival Prediction 

This project builds a Decision Tree classifier using the Titanic dataset and deploys a simple Streamlit web app to predict whether a passenger would survive based on class, sex, age, and fare.

## 🚀 Features

Clean ML pipeline with preprocessing

Decision Tree training & evaluation

Saves model in .pkl and .joblib formats

Interactive Streamlit UI

Integer-only input fields for Age and Fare

Real-time survival prediction

## 📁 Project Structure

  ├── titanic.csv
  
  ├── model_train.py
  
  ├── app.py
  
  ├── titanic_model.joblib
  
  └── titanic_model.pkl

## 🧠 Model Training

To train and save the model:

python model_train.py


## Outputs:

Accuracy score

Confusion matrix

Visual Decision Tree plot

Saved model files

## 🖥️ Run the Streamlit App
streamlit run app.py


This opens the UI in your browser at:

http://localhost:8501

## 🧩 User Inputs

Passenger Class (1, 2, 3)

Sex (Male/Female)

Age (integer only)

Fare (integer only)

## 🔮 Prediction Output

 Survived!

 Did NOT Survive

📦 Dependencies

Install required packages:

pip install pandas numpy scikit-learn matplotlib joblib streamlit
