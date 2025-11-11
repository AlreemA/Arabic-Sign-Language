import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer
from keras.models import Sequential

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Arabic Sign Language Detection", layout="centered")
st.title("Arabic Sign Language Detection Demo")
st.write("You can either capture a hand sign from your webcam or upload a photo.")

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

# ------------------ HELPER FUNCTION ------------------
def preprocess_and_predict(img):
    """Flip, detect hand, crop, resize, normalize, predict."""
    # Flip horizontally
    img = cv2.flip(img, 1)

    # Hand detection using skin mask
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest)
        img_cropped = img[y:y+h_box, x:x+w_box]
    else:
        img_cropped = img

    # Resize & normalize
    img_resized = cv2.resize(img_cropped, (128,128))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_expanded)
    if isinstance(prediction, dict):
        prediction_tensor = list(prediction.values())[0]
    else:
        prediction_tensor = prediction
    prediction_np = prediction_tensor.numpy() if hasattr(prediction_tensor, "numpy") else np.array(prediction_tensor)

    predicted_class = str(np.argmax(prediction_np, axis=-1).item())
    confidence = float(np.max(prediction_np))
    predicted_label = class_name[predicted_class]

    return img_cropped, predicted_label, confidence

st.image(
    "https://i.pinimg.com/originals/e9/a9/93/e9a993a246e099cda75db9116447a281.png",
    caption="Arabic Sign Language Letter Signs",
    use_column_width=True
)


# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["📷 Webcam Demo", "📤 Upload Photo Demo"])

# ------------------ DEMO 1: WEBCAM ------------------
with tab1:
    st.write("Capture a hand sign using your webcam.")
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        img = np.array(Image.open(camera_input))
        # Use preprocessing with flip + crop for webcam
        img_cropped, predicted_label, confidence = preprocess_and_predict(img)
        st.image(img_cropped, caption="Detected Hand Region", use_column_width=True)
        st.markdown(f"### 🧾 Predicted Letter: **{predicted_label}**")
        st.write(f"**Confidence:** {confidence:.2%}")

# ------------------ DEMO 2: UPLOAD IMAGE ------------------
with tab2:
    st.write("Upload a hand sign image for prediction.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file))

        # ------------------ PREPROCESS FOR UPLOAD (NO FLIP / NO CROP) ------------------
        img_resized = cv2.resize(img, (128,128))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # ------------------ PREDICT ------------------
        prediction = model.predict(img_expanded)
        if isinstance(prediction, dict):
            prediction_tensor = list(prediction.values())[0]
        else:
            prediction_tensor = prediction
        prediction_np = prediction_tensor.numpy() if hasattr(prediction_tensor, "numpy") else np.array(prediction_tensor)

        predicted_class = str(np.argmax(prediction_np, axis=-1).item())
        confidence = float(np.max(prediction_np))
        predicted_label = class_name[predicted_class]

        # ------------------ DISPLAY ------------------
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown(f"### 🧾 Predicted Letter: **{predicted_label}**")
        st.write(f"**Confidence:** {confidence:.2%}")
