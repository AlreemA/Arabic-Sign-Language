import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer
from keras.models import Sequential

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
    # Wrap the old SavedModel folder as a Keras layer
    model_layer = TFSMLayer("model2/", call_endpoint='serving_default')
    model = Sequential([model_layer])
    return model

model = load_asl_model()

# ------------------ CAMERA INPUT ------------------
camera_input = st.camera_input("Take a picture")

if camera_input is not None:
    # Convert the image to a NumPy array
    img = Image.open(camera_input)
    img = np.array(img)

    # Preprocess
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_expanded)

    # Convert tensor to numpy
    prediction_np = prediction.numpy() if hasattr(prediction, "numpy") else np.array(prediction)

    # Get predicted class and confidence
    predicted_class = str(np.argmax(prediction_np, axis=-1).item())
    confidence = float(np.max(prediction_np))

    # Get class label
    predicted_label = class_name[predicted_class]

    # Display
    st.image(img, caption="Captured Image", use_column_width=True)
    st.markdown(f"### 🧾 Predicted Letter: **{predicted_label}**")
    st.write(f"**Confidence:** {confidence:.2%}")


