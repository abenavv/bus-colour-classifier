import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('bus_colour_classifier.h5')

# Class names (MUST be in the same order as during training!)
class_names = ['blue', 'green', 'red']

st.title("Kerala Bus Colour Classifier 🎨🚌")

st.write(
    """
    Upload a bus photo and this model will predict whether it's **Blue**, **Green**, or **Red**.
    """
)

uploaded_file = st.file_uploader("Choose a bus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # scale to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # make it (1, 224, 224, 3)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"### Predicted Colour: **{predicted_class}**")
    st.write(f"Prediction Probabilities: {predictions}")

