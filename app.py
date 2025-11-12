import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="🤟 Arabic Sign Language Detection", layout="centered")
st.image("https://i.pinimg.com/originals/e9/a9/93/e9a993a246e099cda75db9116447a281.png", use_column_width=True)
st.title("Arabic Sign Language Detection Demo")

# ------------------ CLASS LABELS ------------------
class_name = {
    "0": "ain", "1": "al", "2": "aleff", "3": "bb", "4": "dal", "5": "dha", "6": "dhad", "7": "fa",
    "8": "gaaf", "9": "ghain", "10": "ha", "11": "haa", "12": "jeem", "13": "kaaf", "14": "khaa",
    "15": "la", "16": "laam", "17": "meem", "18": "nun", "19": "ra", "20": "saad", "21": "seen",
    "22": "sheen", "23": "ta", "24": "taa", "25": "thaa", "26": "thal", "27": "toot", "28": "waw",
    "29": "ya", "30": "yaa", "31": "zay"
}

# ------------------ MODEL DOWNLOAD ------------------
MODEL_PATH = "model2.pth"
MODEL_URL = "https://huggingface.co/your-username/your-model/resolve/main/model2.pth"  # 🔁 change this!

@st.cache_resource
def load_model():
    # Download model if not available
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... this may take a minute.")
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, "wb").write(r.content)
        st.success("Model downloaded successfully!")

    # Define a dummy model (same structure as your trained model)
    class ASLModel(nn.Module):
        def __init__(self, num_classes=32):
            super(ASLModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 32 * 32, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = ASLModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ------------------ IMAGE TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ------------------ DEMOS ------------------
tab1, tab2 = st.tabs(["📸 Camera Demo", "🖼️ Upload Image Demo"])

# ---- CAMERA DEMO ----
with tab1:
    camera_input = st.camera_input("Take a picture")

    if camera_input is not None:
        img = Image.open(camera_input)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally for mirror webcam
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        predicted_label = class_name[str(pred_idx.item())]
        st.image(img, caption=f"Predicted: {predicted_label} ({conf.item()*100:.2f}%)", use_column_width=True)

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
        st.image(img, caption=f"Predicted: {predicted_label} ({conf.item()*100:.2f}%)", use_column_width=True)
