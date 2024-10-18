import streamlit as st
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# Specify which 10 features you want to collect
input_indices = [1, 15, 30, 45, 60, 75, 90, 105, 120, 135]

@st.cache_resource
def create_model():
    # Build the ANN model for binary classification
    model = Sequential()

    # Input layer
    model.add(Dense(256, input_shape=(len(input_indices),), activation='elu'))
    model.add(BatchNormalization())  # Batch Normalization after input layer
    model.add(Dropout(0.2))  # Dropout after Batch Normalization

    # Hidden layers
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(BatchNormalization())  # Batch Normalization after first hidden layer
    model.add(Dropout(0.3))  # Dropout after Batch Normalization

    model.add(Dense(32, activation='elu'))
    model.add(BatchNormalization())  # Batch Normalization after second hidden layer
    model.add(Dropout(0.2))  # Dropout after Batch Normalization

    # Output layer (for binary classification)
    model.add(Dense(1, activation='sigmoid'))  # Output is 1 neuron with sigmoid for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = create_model()

# Title and description
st.title("Epileptic Seizure Recognition")
st.write("This app uses a trained model to predict if a seizure is occurring based on input data.")

# Input function for specific features
def user_input_features():
    features = []
    for idx in input_indices:
        feature_value = st.number_input(f'Feature x{idx}', value=0.0)
        features.append(feature_value)
    input_data = pd.DataFrame([features], columns=[f'Feature x{idx}' for idx in input_indices])
    return input_data

# Collect input data from user
input_data = user_input_features()

# Create a submit button
if st.button('Submit'):
    # Make prediction
    prediction_proba = model.predict(input_data)
    prediction = (prediction_proba > 0.5).astype(int)  # Convert probabilities to class prediction (binary)

    # Display results
    st.write(f"Prediction: {'Seizure' if prediction[0] == 1 else 'No Seizure'}")
    #st.write(f"Probability: {prediction_proba[0][0] * 100:.2f}% seizure, {(1 - prediction_proba[0][0]) * 100:.2f}% no seizure")
