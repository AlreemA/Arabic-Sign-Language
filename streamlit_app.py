import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Arabic Sign Language Detection", layout="centered")
st.title("🤟 Arabic Sign Language Detection Demo")

st.write("Use your webcam to capture a hand sign and predict the Arabic letter.")

# ------------------ CLASS LABELS ------------------
class_name = {
    "0":"ain","1":"al","2":"aleff","3":"bb","4":"dal","5":"dha","6":"dhad","7":"fa",
    "8":"gaaf","9":"ghain","10":"ha","11":"haa","12":"jeem","13":"kaaf","14":"khaa",
    "15":"la","16":"laam","17":"meem","18":"nun","19":"ra","20":"saad","21":"seen",
    "22":"sheen","23":"ta","24":"taa","25":"thaa","26":"thal","27":"toot","28":"waw",
    "29":"ya","30":"yaa","31":"zay"
}

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_asl_model():
    model = tf.keras.models.load_model("model2/")  # Folder containing saved_model.pb
    return model

model = load_asl_model()

# ------------------ CAMERA INPUT ------------------
camera_input = st.camera_input("Take a picture")

if camera_input is not None:
    # Convert the image to a NumPy array
    img = Image.open(camera_input)
    img = np.array(img)

    # Preprocess (match training settings)
    img_resized = cv2.resize(img, (128, 128))  # Match target_size
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_expanded)
    predicted_class = str(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))

    # Get class label
    predicted_label = class_name[predicted_class]

    # Display
    st.image(img, caption="Captured Image", use_column_width=True)
    st.markdown(f"### 🧾 Predicted Letter: **{predicted_label}**")
    st.write(f"**Confidence:** {confidence:.2%}")
