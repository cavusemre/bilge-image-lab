import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Bilge Blur Demo", layout="centered")
st.title("📷 Bilge Blur Demo")
st.write("Bir fotoğraf seç → blur seviyesini ayarla.")

uploaded = st.file_uploader("Fotoğraf yükle (jpg/png)", type=["jpg", "jpeg", "png"])
blur = st.slider("Blur seviyesi", 1, 51, 11, step=2)  # tek sayı olmalı

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    blurred = cv2.GaussianBlur(img_bgr, (blur, blur), 0)
    out_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Orijinal", use_container_width=True)
    st.image(out_rgb, caption="Blur uygulanmış", use_container_width=True)
