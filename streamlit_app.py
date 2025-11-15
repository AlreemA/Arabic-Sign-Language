from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import av
import collections
import time

# ------------------ MODEL ------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 32)
    state_dict = torch.load("src/best_mobilenetv3.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_name = {str(i): label for i, label in enumerate([
    "ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain","ha","haa",
    "jeem","kaaf","khaa","la","laam","meem","nun","ra","saad","seen","sheen",
    "ta","taa","thaa","thal","toot","waw","ya","yaa","zay"
])}

arabic_map = {
    "aleff": "ا","bb": "ب","ta": "ت","jeem": "ج","ha": "ه","ain": "ع","al": "ل","dal": "د",
    "dha": "ذ","dhad": "ض","fa": "ف","gaaf": "ق","ghain": "غ","haa": "ح","kaaf": "ك","khaa": "خ",
    "la": "ل","laam": "ل","meem": "م","nun": "ن","ra": "ر","saad": "ص","seen": "س","sheen": "ش",
    "taa": "ط","thaa": "ث","thal": "ذ","toot": "ط","waw": "و","ya": "ي","yaa": "ي","zay": "ز"
}

# ------------------ HAND CROP ------------------
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Camera Demo",
    "🖼 Upload Demo",
    "🎥 Live Detection",
    "📝 Word Builder",
    "⚡ Word Builder Live"
])

# ---- Initialize session state variables (for all tabs) ----
for key, default in {
    "d5_word": "",
    "d5_predictions": [],
    "d5_confidences": [],
    "d5_countdown_started": False,
    "d5_countdown_start_time": 0.0,
    "d5_camera_open": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default



# ---- CAMERA DEMO ----
with tab1:
    camera_input = st.camera_input("Take a picture")

    if camera_input is not None:
        img = Image.open(camera_input)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_np = np.array(img)

        # Crop hand
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
        img = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        predicted_label = class_name[str(pred_idx.item())]
        st.image(img, caption=f"Predicted: {predicted_label} ({conf.item()*100:.2f}%)", use_container_width=True)


# ---- LIVE DETECTION DEMO ----

with tab3:
    st.write("**Real-time Arabic Sign Language Detection (Webcam Feed)**")
    max_frames = 10
    conf_threshold = 0.85
    consistency_threshold = 0.8
    
    if "live_preds" not in st.session_state:
        st.session_state.live_preds = collections.deque(maxlen=max_frames)
        st.session_state.live_confs = collections.deque(maxlen=max_frames)
    
    def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        cropped = crop_hand(img)
        img_tensor = transform(Image.fromarray(cropped)).unsqueeze(0)
    
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
    
        label = class_name[str(pred_idx.item())]
        conf_val = float(conf.item())
    
        st.session_state.live_preds.append(label)
        st.session_state.live_confs.append(conf_val)
    
        # Consistency check
        most_common = max(set(st.session_state.live_preds), key=st.session_state.live_preds.count)
        freq = st.session_state.live_preds.count(most_common) / len(st.session_state.live_preds)
        avg_conf = sum(st.session_state.live_confs) / len(st.session_state.live_confs)
    
        text = f"{label} ({conf_val*100:.1f}%)"
        if freq >= consistency_threshold and avg_conf >= conf_threshold and len(st.session_state.live_preds) == max_frames:
            text += " ✅"
    
        # Draw text on frame
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        return av.VideoFrame.from_ndarray(img, format="rgb24")
    
    webrtc_streamer(
        key="live_demo",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: type("Processor", (), {"recv": process_frame})()
    )


# ---- WORD BUILDER DEMO (with dynamic adding) ----
with tab4:
    st.write("**Build Words from Hand Sign Images**")

    import hashlib
    from deep_translator import GoogleTranslator  # pip install deep-translator

    # Initialize session state
    for key, default in [
        ("wb_images", []),
        ("wb_letters", []),
        ("wb_filenames", []),
        ("wb_hashes", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # (Optional) guard: if lists ever get out of sync, reset cleanly
    same_len = len(st.session_state.wb_images) == len(st.session_state.wb_letters) == \
               len(st.session_state.wb_filenames) == len(st.session_state.wb_hashes)
    if not same_len:
        st.session_state.wb_images = []
        st.session_state.wb_letters = []
        st.session_state.wb_filenames = []
        st.session_state.wb_hashes = []

    uploaded_files = st.file_uploader(
        "Upload hand sign images (one by one or multiple)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Build NEW lists based on the current uploader contents.
        # Reuse previous predictions if the same content hash exists.
        new_images, new_letters, new_filenames, new_hashes = [], [], [], []

        # Map existing hashes -> (image, letter) so we can reuse
        cache = {
            h: (img, letter)
            for h, img, letter in zip(
                st.session_state.wb_hashes,
                st.session_state.wb_images,
                st.session_state.wb_letters,
            )
        }

        for file in uploaded_files:
            file_bytes = file.read()
            file.seek(0)
            file_hash = hashlib.md5(file_bytes).hexdigest()

            if file_hash in cache:
                # Reuse cached image & letter
                img, letter = cache[file_hash]
            else:
                # New file content: load, predict
                img = Image.open(file).convert("RGB")
                img_cropped = crop_hand(np.array(img))
                img_tensor = transform(Image.fromarray(img_cropped)).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)
                letter = class_name[str(pred_idx.item())]

            new_images.append(img)
            new_letters.append(letter)
            new_filenames.append(file.name)
            new_hashes.append(file_hash)

        # Replace session state with the freshly built lists (no pops, no mismatches)
        st.session_state.wb_images = new_images
        st.session_state.wb_letters = new_letters
        st.session_state.wb_filenames = new_filenames
        st.session_state.wb_hashes = new_hashes
    else:
        # If uploader is empty, clear everything
        st.session_state.wb_images = []
        st.session_state.wb_letters = []
        st.session_state.wb_filenames = []
        st.session_state.wb_hashes = []

    # Display images with their predicted letters
    if st.session_state.wb_images:
        cols = st.columns(len(st.session_state.wb_images))
        for i, img in enumerate(st.session_state.wb_images):
            cols[i].image(img, caption=st.session_state.wb_letters[i])

    arabic_map = {
        "aleff": "ا", "bb": "ب", "ta": "ت", "jeem": "ج", "ha": "ه", "ain": "ع", "al": "ل", "dal": "د",
        "dha": "ذ", "dhad": "ض", "fa": "ف", "gaaf": "ق", "ghain": "غ", "haa": "ح", "kaaf": "ك", "khaa": "خ",
        "la": "ل", "laam": "ل", "meem": "م", "nun": "ن", "ra": "ر", "saad": "ص", "seen": "س", "sheen": "ش",
        "taa": "ط", "thaa": "ث", "thal": "ذ", "toot": "ط", "waw": "و", "ya": "ي", "yaa": "ي", "zay": "ز"
    }

    arabic_word = "".join([arabic_map.get(l, "") for l in st.session_state.wb_letters])
    st.markdown(f"**Arabic Word:** {arabic_word}")

    # ---- TRANSLATION (offline first, then online) ----
    if arabic_word.strip():
        offline_translations = {
            "باب": "door",
            "بيت": "house",
            "قلب": "heart",
            "حب": "love",
            "نور": "light",
            "كتاب": "book",
            "مدرسة": "school",
            "قلم": "pen",
            "ام": "mother",
        }

        translation_text = offline_translations.get(arabic_word)

        if translation_text:
            st.markdown(f"**English Translation (Offline):** {translation_text}")
        else:
            try:
                translation = GoogleTranslator(source="ar", target="en").translate(arabic_word)
                st.markdown(f"**English Translation (Online):** {translation}")
            except Exception:
                st.warning("⚠️ Could not translate automatically. Consider adding this word to the offline dictionary.")

# -------------------- TAB 5: ⚡ WORD BUILDER LIVE --------------------
with tab5:
if "d5_word" not in st.session_state:
    st.session_state.d5_word = ""
    st.session_state.d5_preds = collections.deque(maxlen=max_frames)
    st.session_state.d5_confs = collections.deque(maxlen=max_frames)
    st.session_state.d5_countdown = False
    st.session_state.d5_countdown_start = 0.0

lock_secs = st.slider("Lock-in Duration (s)", 1, 5, 2)

def process_frame_word(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="rgb24")
    cropped = crop_hand(img)
    img_tensor = transform(Image.fromarray(cropped)).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = class_name[str(pred_idx.item())]
    conf_val = float(conf.item())

    st.session_state.d5_preds.append(label)
    st.session_state.d5_confs.append(conf_val)

    most_common = max(set(st.session_state.d5_preds), key=st.session_state.d5_preds.count)
    freq = st.session_state.d5_preds.count(most_common) / len(st.session_state.d5_preds)
    avg_conf = sum(st.session_state.d5_confs) / len(st.session_state.d5_confs)

    # Stability check and countdown
    if len(st.session_state.d5_preds) == max_frames and freq >= consistency_threshold and avg_conf >= conf_threshold:
        if not st.session_state.d5_countdown:
            st.session_state.d5_countdown = True
            st.session_state.d5_countdown_start = time.time()
        elapsed = time.time() - st.session_state.d5_countdown_start
        remaining = lock_secs - elapsed
        if remaining <= 0:
            st.session_state.d5_word += arabic_map.get(most_common, "")
            st.session_state.d5_preds.clear()
            st.session_state.d5_confs.clear()
            st.session_state.d5_countdown = False
    else:
        st.session_state.d5_countdown = False

    # Overlay text
    cv2.putText(img, f"Word: {st.session_state.d5_word}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
    return av.VideoFrame.from_ndarray(img, format="rgb24")

webrtc_streamer(
    key="word_builder_live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: type("ProcessorWord", (), {"recv": process_frame_word})()
)
