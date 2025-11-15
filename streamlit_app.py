from streamlit_webrtc import webrtc_streamer
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import time
import av
import pandas as pd
from datetime import datetime
import os
import collections

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Arabic Sign Language Recognition", layout="centered")
st.image(
    "https://i.pinimg.com/originals/e9/a9/93/e9a993a246e099cda75db9116447a281.png",
    use_container_width=True
)
st.title("Arabic Sign Language Recognition Demos")

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
    # Recreate the exact same architecture
    model = models.mobilenet_v3_large(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 32)

    # Load your trained weights
    state_dict = torch.load("best_mobilenetv3.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ------------------ IMAGE TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # match training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
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
        cropped = img_np[y:y+h, x:x+w]
        return cropped
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
    st.markdown(
    "⚠️ **This demo only works when run locally.** "
    "Please download the repository and follow the instructions in the README file if using cloud deployment link: "
    "[GitHub Repo](https://github.com/AlreemA/Arabic-Sign-Language)",
    unsafe_allow_html=True
)
    run_live = st.checkbox("Start Live Detection")

    FRAME_WINDOW = st.image([])  # video frame placeholder
    status_text = st.empty()     # dynamic status display

    camera = cv2.VideoCapture(0)

    # parameters for stability
    max_frames = 10
    conf_threshold = 0.85
    consistency_threshold = 0.8
    predictions, confidences = [], []

    stop_detected = False
    countdown_started = False
    countdown_start_time = None
    countdown_duration = 5  # seconds

    while run_live and not stop_detected:
        ret, frame = camera.read()
        if not ret:
            st.warning("⚠️ Could not access webcam.")
            break

        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped = crop_hand(frame_rgb)
        cropped_pil = Image.fromarray(cropped).convert("RGB")
        img_tensor = transform(cropped_pil).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        predicted_label = class_name[str(pred_idx.item())]
        predictions.append(predicted_label)
        confidences.append(conf.item())

        if len(predictions) > max_frames:
            predictions.pop(0)
            confidences.pop(0)

        # consistency check
        most_common = max(set(predictions), key=predictions.count)
        freq = predictions.count(most_common) / len(predictions)
        avg_conf = sum(confidences) / len(confidences)

        # show label on frame
        cv2.putText(frame_rgb, f"{predicted_label} ({conf.item()*100:.1f}%)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # stability condition
        if freq >= consistency_threshold and avg_conf >= conf_threshold and len(predictions) == max_frames:
            if not countdown_started:
                countdown_started = True
                countdown_start_time = time.time()
                status_text.warning("🕒 Stable prediction detected! Holding for confirmation...")

            elapsed = time.time() - countdown_start_time
            remaining = countdown_duration - elapsed

            # display countdown overlay
            cv2.putText(frame_rgb, f"Locking in {max(0, int(remaining))}...",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

            # if countdown complete
            if remaining <= 0:
                stop_detected = True
                final_label = most_common
                final_conf = avg_conf
        else:
            countdown_started = False  # reset if unstable
            countdown_start_time = None
            status_text.info("📷 Detecting... please hold your hand steady.")

        FRAME_WINDOW.image(frame_rgb, channels="RGB", width='stretch')

    camera.release()

    if stop_detected:
        status_text.success(f"✅ Final Prediction: **{final_label}** ({final_conf*100:.1f}%)")
    elif run_live:
        status_text.info("🛑 Detection stopped manually.")


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
    st.write("## ⚡ Word Builder Live (Camera-Based)")
    st.markdown(
    "⚠️ **This demo only works when run locally.** "
    "Please download the repository and follow the instructions in the README file if using cloud deployment link: "
    "[GitHub Repo](https://github.com/AlreemA/Arabic-Sign-Language)",
    unsafe_allow_html=True
)
    st.caption("Hold a steady sign; when stable, the letter is added automatically (same logic as Tab 3).")

    # ---- Controls ----
    colA, colB, colC, colD = st.columns([1, 1, 1, 2])
    with colA:
        if st.button("🧹 Clear Word"):
            st.session_state.d5_word = ""
    with colB:
        if st.button("⌫ Backspace"):
            st.session_state.d5_word = st.session_state.d5_word[:-1]
    with colC:
        run_live = st.checkbox("Start Live Word Builder")
    with colD:
        lock_secs = st.slider("Lock-in (s)", 1, 5, 2)
    st.info("Tip: Hold your hand steady until countdown ends.")

    # Parameters
    max_frames = 10
    conf_threshold = 0.85
    consistency_threshold = 0.8

    FRAME_WINDOW = st.image([], use_container_width=True)
    status_text = st.empty()

    # NEW: live placeholders for word & translation
    word_box = st.empty()
    trans_box = st.empty()

    # Arabic letter map
    arabic_map = {
        "aleff": "ا","bb": "ب","ta": "ت","jeem": "ج","ha": "ه","ain": "ع","al": "ل","dal": "د",
        "dha": "ذ","dhad": "ض","fa": "ف","gaaf": "ق","ghain": "غ","haa": "ح","kaaf": "ك","khaa": "خ",
        "la": "ل","laam": "ل","meem": "م","nun": "ن","ra": "ر","saad": "ص","seen": "س","sheen": "ش",
        "taa": "ط","thaa": "ث","thal": "ذ","toot": "ط","waw": "و","ya": "ي","yaa": "ي","zay": "ز"
    }

    # NEW: helper to translate & render live
    def translate_and_render(word: str):
        if not word:
            trans_box.empty()
            return
        variants = {word}
        if word.startswith("ا"):
            variants.update({"أ" + word[1:], "إ" + word[1:], "آ" + word[1:]})
        offline = {
            "باب": "door","بيت": "house","قلب": "heart","حب": "love","نور": "light",
            "كتاب": "book","مدرسة": "school","قلم": "pen","أم": "mother","ام": "mother"
        }
        results, tried_online = [], False
        for form in variants:
            t = offline.get(form)
            if t is None:
                try:
                    from deep_translator import GoogleTranslator
                    t = GoogleTranslator(source="ar", target="en").translate(form)
                    tried_online = True
                except Exception:
                    t = None
            if t:
                results.append((form, t))
        # de-duplicate by English text
        seen, uniq = set(), []
        for form, t in results:
            key = t.strip().lower()
            if key not in seen:
                seen.add(key)
                uniq.append((form, t))
        if uniq:
            trans_box.markdown("**English Translation (candidates):**\n" +
                               "\n".join([f"- {f} → {t}" for f, t in sorted(uniq, key=lambda x: 0 if x[0]==word else 1)]))
        else:
            trans_box.warning("⚠️ Translation unavailable." if tried_online else "⚠️ Offline only. Try connecting to the internet.")

    # Camera open once
    camera = None
    if run_live:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            status_text.error("⚠️ Could not access webcam. Check permissions or close other apps.")
            run_live = False

    import collections, time
    predictions, confidences = collections.deque(maxlen=max_frames), collections.deque(maxlen=max_frames)
    countdown_started, countdown_start_time = False, 0.0

    while run_live:
        ret, frame = camera.read()
        if not ret:
            status_text.warning("⚠️ Lost camera connection.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            cropped = crop_hand(frame_rgb)
        except Exception:
            cropped = frame_rgb

        img_tensor = transform(Image.fromarray(cropped)).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        predicted_label = class_name[str(pred_idx.item())]
        conf_val = float(conf.item())

        predictions.append(predicted_label)
        confidences.append(conf_val)

        # stability check
        most_common = max(set(predictions), key=predictions.count)
        freq = predictions.count(most_common) / len(predictions)
        avg_conf = sum(confidences) / len(confidences)

        cv2.putText(frame_rgb, f"{predicted_label} ({conf_val*100:.1f}%)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,150,255), 2)

        stable = (len(predictions) == max_frames and
                  freq >= consistency_threshold and
                  avg_conf >= conf_threshold)

        if stable and not countdown_started:
            countdown_started = True
            countdown_start_time = time.time()
            status_text.warning("🕒 Stable sign detected… holding!")

        if countdown_started:
            remaining = lock_secs - (time.time() - countdown_start_time)
            if remaining > 0:
                cv2.putText(frame_rgb, f"Locking in {int(remaining)}s...",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,0), 2)
            else:
                # Add the locked letter
                arabic_char = arabic_map.get(most_common, "")
                if arabic_char:
                    st.session_state.d5_word += arabic_char
                    status_text.success(f"✅ Added: {arabic_char} ({most_common})")
                    # NEW: update live word & translation immediately
                    word_box.markdown(f"### 📝 Arabic Word (Live): `{st.session_state.d5_word}`")
                    translate_and_render(st.session_state.d5_word)
                predictions.clear()
                confidences.clear()
                countdown_started = False

        # Draw overlays and show frame
        cv2.putText(frame_rgb, f"Word: {st.session_state.d5_word}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
        time.sleep(0.02)

    if camera:
        camera.release()
        status_text.info("🛑 Camera released.")

    # Also render once when camera is OFF (so you still see latest translation)
    current_word = st.session_state.d5_word
    word_box.markdown(f"### 📝 Arabic Word (Live): `{current_word or '—'}`")
    translate_and_render(current_word)
