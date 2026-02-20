import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Image Classifier", layout="centered")

# Load the trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.h5')

model = load_model()

# Define the image size expected by your model
IMG_SIZE = (224, 224)

# Define class names (replace with your actual class labels)
class_names = ['Class A', 'Class B', 'Class C', 'Class D']  # <-- customize this

# Streamlit app UI
st.title("🧠 Image Classifier")
st.write("Upload an image, and the model will predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize (standard practice for CNNs)

    # Make prediction
    prediction = model.predict(img_array)

    # Get predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.success(f"Prediction: {class_names[predicted_class]} ({confidence * 100:.2f}% confidence)")
