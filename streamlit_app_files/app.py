import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Define label names
labels = [
    'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60',
    'Speed Limit 70', 'Speed Limit 80', 'End of Speed Limit 80', 'Speed Limit 100',
    'Speed Limit 120', 'No Passing', 'No Passing Veh Over 3.5 Tons', 'Right of Way at Intersection',
    'Priority Road', 'Yield', 'Stop', 'No Vehicles', 'Veh Over 3.5 Tons Prohibited',
    'No Entry', 'General Caution', 'Dangerous Curve Left', 'Dangerous Curve Right',
    'Double Curve', 'Bumpy Road', 'Slippery Road', 'Road Narrows on The Right',
    'Road Work', 'Traffic Signals', 'Pedestrians', 'Children Crossing',
    'Bicycles Crossing', 'Beware of Ice Snow', 'Wild Animals Crossing',
    'End Speed Passing Limits', 'Turn Right Ahead', 'Turn Left Ahead',
    'Ahead Only', 'Go Straight or Right', 'Go Straight or Left',
    'Keep Right', 'Keep Left', 'Roundabout Mandatory', 'End of Nopassing',
    'End Nopassing Veh Over 3.5 Tons'
]

# Load the model
model = load_model('traffic_signs_model.h5')

# Add custom CSS
st.markdown("""
<style>
    body {
        background-color: #f0f2f5;
    }
    .title {
        text-align: center;
        color: #FF6347;
        font-size: 48px;
        font-weight: bold;
        margin: 20px 0;
    }
    .description {
        text-align: center;
        font-size: 24px;
        margin-bottom: 30px;
        color: #555;
    }
    .result {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #4CAF50;
        margin: 20px 0;
    }
    .uploaded-image {
        border: 2px solid #FF6347;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="title">🚦 Traffic Sign Prediction App 🚦</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload an image of a traffic sign to see the model\'s prediction.</p>', unsafe_allow_html=True)

# Image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process the image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape to (1, 64, 64, 3)

    # Make prediction
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)

    # Show the result
    st.markdown(f'<div class="result">Predicted Label: **{labels[predicted_label]}**</div>', unsafe_allow_html=True)
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True, output_format='auto', clamp=True)

# Footer
st.markdown('<div class="footer">Developed by comeis | 🚀 Let\'s make the roads safer!</div>', unsafe_allow_html=True)