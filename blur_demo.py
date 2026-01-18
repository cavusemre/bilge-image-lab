import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="centered")
st.title("🧒 Face Blur Demo – MVP 0.2")
st.write("Fotoğraf yükle → yüz bul → yüzü otomatik blurla. (Cloud uyumlu)")

uploaded = st.file_uploader("Bir fotoğraf seç (JPG/PNG)", type=["jpg", "jpeg", "png"])

min_size = st.slider("Min yüz boyutu", 10, 150, 30)
scale_factor = st.slider("scaleFactor", 101, 130, 110) / 100.0   # 1.01–1.30
min_neighbors = st.slider("minNeighbors", 1, 12, 4)
blur_k = st.slider("Blur gücü (tek sayı)", 11, 99, 31, step=2)
show_boxes = st.checkbox("Yüz kutularını göster", value=True)

if uploaded is None:
    st.info("Yukarıdan foto yükle.")
    st.stop()

# PIL -> OpenCV
img_pil = Image.open(uploaded).convert("RGB")
img = np.array(img_pil)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Face detector (OpenCV built-in path)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=scale_factor,
    minNeighbors=min_neighbors,
    minSize=(min_size, min_size),
)

st.write(f"Bulunan yüz sayısı: **{len(faces)}**")

out = img_bgr.copy()

for (x, y, w, h) in faces:
    # blur face ROI
    roi = out[y:y+h, x:x+w]
    if roi.size == 0:
        continue
    roi_blur = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
    out[y:y+h, x:x+w] = roi_blur

    if show_boxes:
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)

# OpenCV -> RGB for display
out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

st.image(img, caption="Orijinal", use_container_width=True)
st.image(out_rgb, caption="Yüz Blur'lu", use_container_width=True)
