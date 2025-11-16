import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load trained model
model_path = "mobilenetv2_stress_detection.h5"
model = load_model(model_path)

# Labels for binary classification
labels = ["Not Stressed", "Stressed"]

# Streamlit App Configuration 
st.set_page_config(page_title="Facial Stress Detection", page_icon="😟", layout="centered")

st.title("😟 Facial Stress Detection")
st.markdown("Upload a facial image to detect **stress level** (Stressed / Not Stressed).")

# Upload Image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

# Prediction Function
def predict_image(image):
    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize + normalize
    img_resized = cv2.resize(img_cv, (96, 96))
    img_normalized = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict
    pred = model.predict(img_input)[0][0]
    label = labels[1] if pred >= 0.5 else labels[0]
    confidence = pred if pred >= 0.5 else 1 - pred

    return label, confidence

# Run Prediction
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # st.write("🔍 Detection stress level...")
    # label, confidence = predict_image(image)

    # Predict Button
    if st.button(" Predict Stress Level"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(image)

        # Display result
        st.markdown("---")
        if label == "Stressed":
            st.error(f"😰 Prediction: **{label}** (Confidence: {confidence:.2f})")
        else:
            st.success(f"😊 Prediction: **{label}** (Confidence: {confidence:.2f})")
