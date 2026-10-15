
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

st.set_page_config(
    page_title="Fruit & Vegetable Recognition",
    page_icon="🥦"
)

st.title("🍎 Fruit & Vegetable Recognition")
st.write("Upload an image to identify the fruit or vegetable.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Identify Image"):
        with st.spinner("Analyzing image..."):
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array, verbose=0)
            results = decode_predictions(predictions, top=5)[0]

        st.subheader("Prediction Results")

        for _, label, score in results:
            st.write(f"**{label.replace('_', ' ').title()}**")
            st.progress(float(score))
            st.write(f"Confidence: {score * 100:.2f}%")
