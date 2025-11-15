from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import collections
import time
from deep_translator import GoogleTranslator
import hashlib

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

arabic_map = {
    "aleff": "ا","bb": "ب","ta": "ت","jeem": "ج","ha": "ه","ain": "ع","al": "ل","dal": "د",
    "dha": "ذ","dhad": "ض","fa": "ف","gaaf": "ق","ghain": "غ","haa": "ح","kaaf": "ك","khaa": "خ",
    "la": "ل","laam": "ل","meem": "م","nun": "ن","ra": "ر","saad": "ص","seen": "س","sheen": "ش",
    "taa": "ط","thaa": "ث","thal": "ذ","toot": "ط","waw": "و","ya": "ي","yaa": "ي","zay": "ز"
}

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 32)
    state_dict = torch.load("best_mobilenetv3.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ------------------ IMAGE TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------ HAND CROP FUNCTION ------------------
def crop_hand(img_np):
    import cv2
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
        cropped = img_np[y:y+h, x:x+w]
        return cropped
    return img_np

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Camera Demo",
    "🖼 Upload Demo",
    "🎥 Live Detection",
    "📝 Word Builder",
    "⚡ Word Builder Live"
])

# ---- Initialize session state ----
for key, default in {
    "d5_word": "",
    "wb_images": [],
    "wb_letters": [],
    "wb_filenames": [],
    "wb_hashes": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------ TAB 1: Camera Demo ------------------
with tab1:
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        img = Image.open(camera_input).transpose(Image.FLIP_LEFT_RIGHT)
        img_np = np.array(img)
        img_cropped = crop_hand(img_np)
        img_tensor = transform(Image.fromarray(img_cropped)).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        predicted_label = class_name[str(pred_idx.item())]
        st.image(img_cropped, caption=f"Predicted: {predicted_label} ({conf.item()*100:.2f}%)",
                 use_container_width=True)

# ------------------ TAB 2: Upload Demo ------------------
with tab2:
    uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        predicted_label = class_name[str(pred_idx.item())]
        st.image(img, caption=f"Predicted: {predicted_label} ({conf.item()*100:.2f}%)",
                 use_container_width=True)

# ------------------ TAB 3: Live Detection ------------------
with tab3:
    st.write("**Real-time Arabic Sign Language Detection (Webcam)**")
    max_frames = 10
    conf_threshold = 0.85
    consistency_threshold = 0.8
    countdown_duration = 5

    class LiveDetectionProcessor(VideoProcessorBase):
        def __init__(self):
            self.predictions = collections.deque(maxlen=max_frames)
            self.confidences = collections.deque(maxlen=max_frames)
            self.countdown_started = False
            self.countdown_start_time = 0.0
            self.stop_detected = False

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_cropped = crop_hand(img_rgb)
            img_tensor = transform(Image.fromarray(img_cropped)).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            label = class_name[str(pred_idx.item())]
            conf_val = float(conf.item())

            self.predictions.append(label)
            self.confidences.append(conf_val)

            # Stability check
            most_common = max(set(self.predictions), key=self.predictions.count)
            freq = self.predictions.count(most_common)/len(self.predictions)
            avg_conf = sum(self.confidences)/len(self.confidences)

            if freq>=consistency_threshold and avg_conf>=conf_threshold and len(self.predictions)==max_frames:
                if not self.countdown_started:
                    self.countdown_started = True
                    self.countdown_start_time = time.time()
                elapsed = time.time()-self.countdown_start_time
                remaining = countdown_duration - elapsed
                cv2.putText(img, f"Locking in {max(0,int(remaining))}s",
                            (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                if remaining<=0:
                    st.session_state.final_label = most_common
                    st.session_state.final_conf = avg_conf
                    self.predictions.clear()
                    self.confidences.clear()
                    self.countdown_started=False
            else:
                self.countdown_started=False

            cv2.putText(img, f"{label} ({conf_val*100:.1f}%)", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="live-detection",
        video_processor_factory=LiveDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if "final_label" in st.session_state:
        st.success(f"✅ Final Prediction: {st.session_state.final_label} "
                   f"({st.session_state.final_conf*100:.1f}%)")

# ------------------ TAB 4: Word Builder ------------------
# (Same as your previous implementation with file uploader, session state caching, translation)

# ------------------ TAB 5: Word Builder Live ------------------
with tab5:
    st.write("## ⚡ Word Builder Live (Camera-Based)")
    lock_secs = st.slider("Lock-in seconds", 1,5,2)
    max_frames = 10
    conf_threshold = 0.85
    consistency_threshold = 0.8

    class WordBuilderProcessor(VideoProcessorBase):
        def __init__(self):
            self.predictions = collections.deque(maxlen=max_frames)
            self.confidences = collections.deque(maxlen=max_frames)
            self.countdown_started = False
            self.countdown_start_time = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_cropped = crop_hand(img_rgb)
            img_tensor = transform(Image.fromarray(img_cropped)).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            label = class_name[str(pred_idx.item())]
            conf_val = float(conf.item())
            self.predictions.append(label)
            self.confidences.append(conf_val)

            most_common = max(set(self.predictions), key=self.predictions.count)
            freq = self.predictions.count(most_common)/len(self.predictions)
            avg_conf = sum(self.confidences)/len(self.confidences)
            stable = len(self.predictions)==max_frames and freq>=consistency_threshold and avg_conf>=conf_threshold

            if stable and not self.countdown_started:
                self.countdown_started = True
                self.countdown_start_time = time.time()

            if self.countdown_started:
                remaining = lock_secs - (time.time()-self.countdown_start_time)
                cv2.putText(img, f"Locking in {max(0,int(remaining))}s...",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
                if remaining<=0:
                    arabic_char = arabic_map.get(most_common,"")
                    st.session_state.d5_word += arabic_char
                    self.predictions.clear()
                    self.confidences.clear()
                    self.countdown_started=False

            cv2.putText(img, f"Word: {st.session_state.d5_word}",(20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx2 = webrtc_streamer(
        key="word-builder-live",
        video_processor_factory=WordBuilderProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown(f"### 📝 Current Word: `{st.session_state.d5_word}`")
