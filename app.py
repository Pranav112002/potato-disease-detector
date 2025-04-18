import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("model/potato_disease_model.h5")

# Class labels (make sure this matches the training label order!)
class_labels = ['Early Blight', 'Late Blight', 'Healthy']

# Disease information
disease_info = {
    "Early Blight": {
        "description": "Fungal disease caused by *Alternaria solani*. Appears as dark spots with concentric rings on older leaves.",
        "prescription": "🌿 Remove infected leaves.\n🧪 Apply fungicides like mancozeb or chlorothalonil every 7–10 days during early stages."
    },
    "Late Blight": {
        "description": "Caused by *Phytophthora infestans*. Leads to large, dark brown lesions with a yellow halo.",
        "prescription": "🛑 Remove infected plants immediately.\n🧪 Use copper-based fungicides.\n💧 Avoid overhead watering."
    },
    "Healthy": {
        "description": "Your plant appears healthy! No visible signs of disease.",
        "prescription": "✅ Continue good care.\n💧 Water in the morning.\n🌞 Ensure sunlight and airflow."
    }
}

# Set page config
st.set_page_config(page_title="Potato Plant Disease Detection", layout="centered")

# App title
st.title("🥔 Potato Disease Detection App")
st.markdown("Upload a leaf image to detect **Early Blight**, **Late Blight**, or confirm if it's **Healthy**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))  # match training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]

    # Display result
    st.success(f"**Prediction: {predicted_label}**")
    st.markdown(f"🩺 **Disease Info**: {disease_info[predicted_label]['description']}")
    st.markdown(f"💊 **Prescription**: {disease_info[predicted_label]['prescription']}")
