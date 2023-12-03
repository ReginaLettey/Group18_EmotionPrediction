






























import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
# Assuming you have a Pickle file named 'data.pkl'
file_path = "C:/Users/lette/OneDrive/Desktop/EmotionPredSong/EmotionPredSong.pkl"

# Load data from the Pickle file
with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)
    
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Now, 'loaded_data' contains the deserialized data from the Pickle file
print(loaded_data)


# Define emotion labels
emotion_labels = {0: 'sad', 1: 'happy', 2: 'energetic', 3: 'calm'}

# Streamlit app
st.title("Emotion Prediction from Song Features")

# Define a function for making predictions
def predict_emotion(features):
    # Perform prediction using the loaded model
    predictions = loaded_data.predict(features)
    return predictions

# Create a form for user input
st.sidebar.header("User Input")


feature_names = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'uri']

# Initialize a dictionary to store user input
user_input = {}

# Create input fields for each feature
for feature_name in feature_names:
    user_input[feature_name] = st.sidebar.slider(f"Enter {feature_name}:", min_value=0.0, max_value=10.0, value=0.5)

# Convert user input to a NumPy array
user_input_array = np.array([user_input[feature_name] for feature_name in feature_names]).reshape(1, -1, 1)

# Make predictions when the user clicks the "Predict" button
if st.sidebar.button("Predict"):
    predictions = predict_emotion(user_input_array)

    # Display the predicted probabilities
    st.subheader("Predicted Emotion Probabilities:")
    st.write(predictions)

    # Get the predicted class (index with the highest probability)
    predicted_class = np.argmax(predictions)

    # Map the predicted class to emotion label
    predicted_emotion = emotion_labels.get(predicted_class, "Unknown Emotion")

    # Display the predicted emotion
    st.subheader("Predicted Emotion:")
    st.write(predicted_emotion)