# streamlit_asl_webrtc_app.py
# Rewritten Streamlit app using streamlit-webrtc for webcam support (works on Streamlit Cloud)
# Place your trained weights `best_mobilenetv3.pth` in the same repository (repo root or adjust MODEL_PATH).
# requirements.txt should include at least:
# streamlit
# streamlit-webrtc
# torch
# torchvision
# opencv-python-headless
# Pillow
# numpy
# pandas
# deep-translator
# av

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import time
from av import VideoFrame

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Arabic Sign Language Detection (WebRTC)", layout="centered")
st.title("Arabic Sign Language Detection — WebRTC version")

MODEL_PATH = "best_mobilenetv3.pth"  # put this file in your repo
DEVICE = "cpu"

# ------------------ LABELS ------------------
class_name = {
    "0": "ain", "1": "al", "2": "aleff", "3": "bb", "4": "dal", "5": "dha", "6": "dhad", "7": "fa",
    "8": "gaaf", "9": "ghain", "10": "ha", "11": "haa", "12": "jeem", "13": "kaaf", "14": "khaa",
    "15": "la", "16": "laam", "17": "meem", "18": "nun", "19": "ra", "20": "saad", "21": "seen",
    "22": "sheen", "23": "ta", "24": "taa", "25": "thaa", "26": "thal", "27": "toot", "28": "waw",
    "29": "ya", "30": "yaa", "31": "zay"
}

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    model = models.mobilenet_v3_large(pretrained=False)
    # Some torchvision versions use index 3 for classifier final linear layer; guard for both variants
    if isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 4:
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(class_name))
    else:
        # fallback if structure differs
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            last_idx = len(model.classifier) - 1
            in_features = model.classifier[last_idx].in_features
            model.classifier[last_idx] = nn.Linear(in_features, len(class_name))

    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error(f"Model file not found at '{path}'. Please upload it to the repo.")
        raise
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

    model.to(DEVICE)
    model.eval()
    return model

# load once
MODEL = None
try:
    MODEL = load_model()
except Exception:
    st.stop()

# ------------------ TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------ HAND CROP ------------------
def crop_hand(img_rgb: np.ndarray) -> np.ndarray:
    # img_rgb expected in RGB order
    try:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    except Exception:
        return img_rgb
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
        # sanity check
        h_img, w_img = img_rgb.shape[:2]
        if w > 10 and h > 10:
            x1 = max(0, x); y1 = max(0, y); x2 = min(w_img, x+w); y2 = min(h_img, y+h)
            cropped = img_rgb[y1:y2, x1:x2]
            if cropped.size == 0:
                return img_rgb
            return cropped
    return img_rgb

# ------------------ VIDEO PROCESSOR ------------------
class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.max_frames = 8
        self.predictions = []
        self.confidences = []
        self.conf_threshold = 0.75
        self.consistency_threshold = 0.7

    def recv(self, frame: VideoFrame) -> VideoFrame:
        # convert to numpy BGR
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # crop & preprocess
        cropped = crop_hand(img_rgb)
        pil = Image.fromarray(cropped).convert("RGB")
        try:
            img_t = transform(pil).unsqueeze(0)
        except Exception:
            img_t = transform(Image.fromarray(img_rgb)).unsqueeze(0)

        # predict
        with torch.no_grad():
            outputs = MODEL(img_t.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        pred_label = class_name[str(pred_idx.item())]
        conf_val = float(conf.item())

        # rolling buffers
        self.predictions.append(pred_label)
        self.confidences.append(conf_val)
        if len(self.predictions) > self.max_frames:
            self.predictions.pop(0)
            self.confidences.pop(0)

        most_common = max(set(self.predictions), key=self.predictions.count)
        freq = self.predictions.count(most_common) / len(self.predictions)
        avg_conf = sum(self.confidences) / len(self.confidences)

        # overlay info
        display_text = f"{pred_label} ({conf_val*100:.1f}%)"
        cv2.putText(img_rgb, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(img_rgb, f"Stability: {freq:.2f}  AvgConf: {avg_conf:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        # convert back to BGR for streaming
        out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        new_frame = VideoFrame.from_ndarray(out_bgr, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# ------------------ UI LAYOUT ------------------
st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["Camera (WebRTC)", "Upload Image", "Camera (static)"])

if mode == "Camera (WebRTC)":
    st.write("### Use your browser camera (works on Streamlit Cloud)")

    # optional RTC configuration (only include if you need STUN/TURN servers)
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_ctx = webrtc_streamer(
        key="asl-webrtc",
        video_processor_factory=ASLVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

    st.markdown("---")
    st.info("Tip: Hold your hand in front of the camera; the processor will highlight predicted label and stability metrics.")

elif mode == "Upload Image":
    st.write("### Upload an image file for single-frame prediction")
    uploaded = st.file_uploader("Upload a hand sign image", type=["jpg","jpeg","png"]) 
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        cropped = crop_hand(img_np)
        st.image(cropped, caption="Cropped input (what model sees)", use_container_width=True)
        img_t = transform(Image.fromarray(cropped)).unsqueeze(0)
        with torch.no_grad():
            outputs = MODEL(img_t.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        pred_label = class_name[str(pred_idx.item())]
        st.success(f"Prediction: {pred_label} ({conf.item()*100:.2f}%)")

else:
    # static camera via st.camera_input (for local testing in browser)
    st.write("### Static camera capture (useful for local testing)")
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        img = Image.open(camera_input).convert("RGB")
        img_np = np.array(img)
        cropped = crop_hand(img_np)
        st.image(cropped, caption="Cropped input (what model sees)", use_container_width=True)
        img_t = transform(Image.fromarray(cropped)).unsqueeze(0)
        with torch.no_grad():
            outputs = MODEL(img_t.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        pred_label = class_name[str(pred_idx.item())]
        st.success(f"Prediction: {pred_label} ({conf.item()*100:.2f}%)")

# Footer notes
st.markdown("---")
st.caption("Notes:\n- This app uses streamlit-webrtc for browser webcam access (compatible with Streamlit Cloud).\n- Ensure best_mobilenetv3.pth is present in the repo root.\n- If you run into package build errors on Streamlit Cloud, prefer opencv-python-headless and pin av/opencv versions in requirements.txt.")
