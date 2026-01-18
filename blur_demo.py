import io
import numpy as np
import cv2
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Bilge Blur Demo", page_icon="📷", layout="centered")

st.title("📷 Bilge Blur Demo")
st.caption("Bir fotoğraf seç → blur seviyesini ayarla → sonucu indir.")

uploaded = st.file_uploader("Fotoğraf yükle (jpg/png)", type=["jpg", "jpeg", "png"])

blur = st.slider("Blur seviyesi", 1, 51, 11, step=2)  # tek sayılar daha iyi

def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    rgb = np.array(img_pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    img_cv = pil_to_cv(img_pil)

    blurred_cv = cv2.GaussianBlur(img_cv, (blur, blur), 0)
    blurred_pil = cv_to_pil(blurred_cv)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Önce")
        st.image(img_pil, use_container_width=True)

    with col2:
        st.subheader("Sonra")
        st.image(blurred_pil, use_container_width=True)

    # download
    buf = io.BytesIO()
    blurred_pil.save(buf, format="PNG")
    st.download_button(
        "⬇️ Blur’lu görseli indir (PNG)",
        data=buf.getvalue(),
        file_name="bilge_blur.png",
        mime="image/png"
    )
else:
    st.info("Yukarıdan bir fotoğraf yükle. Sonuç sağda görünecek.")
