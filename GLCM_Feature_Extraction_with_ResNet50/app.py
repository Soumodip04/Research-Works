# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.layers import Flatten, Input 
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
import joblib
from utils import prepare_image

st.title("Breast Cancer Image Classifier")
st.write("Upload a grayscale ultrasound image to classify it as **Benign**, **Malignant**, or **Normal**.")

# Load ResNet50 for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in base_model.layers[:-20]:
    layer.trainable = False
resnet_model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

# Load MLP model
model = load_model("model/model_weights.h5")

# Load scaler
scaler = joblib.load("model/scaler.pkl")  # Save using joblib.dump(scaler, 'model/scaler.pkl')

# Class names
classes = ['benign', 'malignant', 'normal']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Feature extraction
    features, _ = prepare_image(uploaded_file, resnet_model)
    features_scaled = scaler.transform(features)

    # Prediction
    pred_probs = model.predict(features_scaled)[0]
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[pred_class] * 100

    st.markdown(f"### 🧠 Prediction: **{classes[pred_class].capitalize()}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
