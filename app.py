import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Arabic Sign Language Detection", layout="centered")
st.image(
    "https://i.pinimg.com/originals/e9/a9/93/e9a993a246e099cda75db9116447a281.png",
    use_container_width=True
)
st.title("Arabic Sign Language Detection Demo")

# ------------------ CLASS LABELS ------------------
class_name = {
    "0": "ain", "1": "al", "2": "aleff", "3": "bb", "4": "dal", "5": "dha", "6": "dhad", "7": "fa",
    "8": "gaaf", "9": "ghain", "10": "ha", "11": "haa", "12": "jeem", "13": "kaaf", "14": "khaa",
    "15": "la", "16": "laam", "17": "meem", "18": "nun", "19": "ra", "20": "saad", "21": "seen",
    "22": "sheen", "23": "ta", "24": "taa", "25": "thaa", "26": "thal", "27": "toot", "28": "waw",
    "29": "ya", "30": "yaa", "31": "zay"
}

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model():
    # Load MobileNetV3 small backbone
    model = models.mobilenet_v3_small(weights=None)
    # Replace classifier with correct number of classes
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 32)
    # Load your checkpoint
    state_dict = torch.load("best_mobilenetv3.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ------------------ IMAGE TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ------------------ HAND CROP FUNCTION ------------------
def crop_hand(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return img_np[y:y+h, x:x+w]
    return img_np

# ------------------ DEMOS ------------------
tab1, tab2 = st.tabs(["Camera Demo", "Upload Image Demo"])

# ---- CAMERA DEMO ----
with tab1:
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        img = Image.open(camera_input)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_np = np.array(img)
        img_cropped = crop_hand(img_np)
        img_tensor = transform(Image.fromarray(img_cropped)).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        predicted_label = class_name[str(pred_idx.item())]
        st.image(img_cropped, caption=f"Predicted: {predicted_label} ({conf.item()*100:.2f}%)", use_container_width=True)

# ---- UPLOAD DEMO ----
with tab2:
    uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        predicted_label = class_name[str(pred_idx.item())]
        st.image(img, caption=f"Predicted: {predicted_label} ({conf.item()*100:.2f}%)", use_container_width=True)
