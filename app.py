import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

st.title("🧠 YOLOv8 Object Detection")

# Load model
model = YOLO("model.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img.save(tmp_file.name)
        results = model(tmp_file.name)

    # Show results
    for r in results:
        result_img = r.plot()  # numpy image
        st.image(result_img, caption="Detection Result", use_column_width=True)
