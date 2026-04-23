import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Setup
st.set_page_config(page_title="Labor Safety AI", layout="centered")

@st.cache_resource
def load_model():
    # Downloads the AI model automatically on first run
    return YOLO('yolov8n.pt') 

model = load_model()

st.title("👷 Safety Compliance Portal")

# Camera Input for Mobile
img_file = st.camera_input("Take a photo to verify PPE")

if img_file:
    img = Image.open(img_file)
    results = model.predict(np.array(img), conf=0.4)
    
    # Show the image with boxes around people
    st.image(results[0].plot(), caption="AI Scan Complete")
    
    # Check for 'person' class (index 0 in COCO)
    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    if 'person' in labels:
        st.success(f"Verified: {labels.count('person')} Workers Active")
    else:
        st.warning("No personnel detected in frame.")
