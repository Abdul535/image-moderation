import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

# Load your pre-trained model
model = load_model()

# Streamlit interface
st.title('Image Classification App')
st.header('Identify if an image is "Violent", "Adult Content", or "Safe"')

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Function to preprocess the image
    def preprocess_image(image, target_size):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        return image

    # Image processing and prediction
    target_size = (224, 224)  # Change based on your model's input size
    processed_image = preprocess_step(image, target_size)
    prediction = model.predict(processed_image)
    labels = ['Violent', 'Adult Content', 'Safe']
    st.write(f"Prediction: {labels[np.argmax(prediction)]}")

# Run this script using: streamlit run app.py
