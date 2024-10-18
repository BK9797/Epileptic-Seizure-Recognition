import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model():
    # Load the model once and reuse it across sessions
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()


# Title and description
st.title("Epileptic Seizure Recognition")
st.write("This app uses a trained model to predict if a seizure is occurring based on input data.")

# Input function
def user_input_features():
    n_features = 178  # Assuming you have 178 input features
    features = [st.number_input(f'Feature {i+1}', value=0.0) for i in range(n_features)]
    input_data = pd.DataFrame([features], columns=[f'Feature {i+1}' for i in range(n_features)])
    return input_data

# Collect input data from user
input_data = user_input_features()

# Make prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display results
st.write(f"Prediction: {'Seizure' if prediction[0] == 1 else 'No Seizure'}")
st.write(f"Probability: {prediction_proba[0][1] * 100:.2f}% seizure, {prediction_proba[0][0] * 100:.2f}% no seizure")

