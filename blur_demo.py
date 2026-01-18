import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Bilge Privacy Lab", page_icon="🛡️", layout="wide")

st.title("🛡️ Bilge Privacy Lab")
st.caption("Fotoğraf yükle → yüz bul → gizle → indir (Cloud uyumlu)")

# --- Cascade yükle (repo içinden) ---
CASCADE_PATH = Path(__file__).parent / "assets" / "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_cascade.empty():
    st.error("Cascade dosyası yüklenemedi. assets/haarcascade_frontalface_default.xml repoda olmalı.")
    st.stop()

# --- Sol panel kontroller ---
st.sidebar.header("Kontroller")

child_mode = st.sidebar.toggle("👶 Child Privacy Mode (daha agresif)", value=True)
show_boxes = st.sidebar.toggle("🟩 Yüz kutularını göster", value=False)
bg_blur_mode = st.sidebar.toggle("🌫️ Arka plan blur (yüz net)", value=False)

mask_type = st.sidebar.selectbox("Yüz gizleme tipi", ["Gaussian", "Pixelate"], index=0)

min_face = st.sidebar.slider("Min yüz boyutu", 20, 120, 30, 1)
scale_factor = st.sidebar.slider("scaleFactor", 105, 150, 120, 1) / 100.0  # 1.05 - 1.50
min_neighbors = st.sidebar.slider("minNeighbors", 3, 12, 4, 1)

blur_strength = st.sidebar.slider("Blur gücü (tek sayı)", 7, 61, 31, 2)  # tek sayı
pixel_size = st.sidebar.slider("Pixel boyutu", 6, 40, 14, 1)

# Child mode agresif ayar (otomatik güçlendirme)
if child_mode:
    blur_strength = max(blur_strength, 41)
    min_face = min(min_face, 30)
    min_neighbors = max(min_neighbors, 4)

# --- Upload ---
uploaded = st.file_uploader("📤 Fotoğraf yükle (JPG/PNG)", type=["jpg", "jpeg", "png"])

def pixelate_roi(roi, block_size=14):
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return roi
    small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_face_hide(img_bgr, faces):
    out = img_bgr.copy()

    # Arka plan blur modu: önce tüm görüntüyü blurla, sonra yüzleri orijinalden geri koy
    if bg_blur_mode:
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred_all = cv2.GaussianBlur(out, (k, k), 0)
        # yüzleri net bırakacağız
        for (x, y, w, h) in faces:
            blurred_all[y:y+h, x:x+w] = out[y:y+h, x:x+w]
        out = blurred_all
        return out

    # Normal: yüzleri gizle
    for (x, y, w, h) in faces:
        roi = out[y:y+h, x:x+w]
        if mask_type == "Gaussian":
            k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            roi2 = cv2.GaussianBlur(roi, (k, k), 0)
        else:
            roi2 = pixelate_roi(roi, pixel_size)
        out[y:y+h, x:x+w] = roi2
    return out

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face, min_face)
    )

    st.write(f"✅ Bulunan yüz sayısı: **{len(faces)}**")

    out_bgr = apply_face_hide(img_bgr, faces)

    # kutu gösterme (debug)
    if show_boxes:
        for (x, y, w, h) in faces:
            cv2.rectangle(out_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orijinal")
        st.image(img_rgb, use_column_width=True)
    with col2:
        st.subheader("Gizlenmiş")
        st.image(out_rgb, use_column_width=True)

    # --- İNDİRME BUTONU ---
    out_pil = Image.fromarray(out_rgb)
    import io
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button(
        "⬇️ Gizlenmiş fotoğrafı indir (PNG)",
        data=buf.getvalue(),
        file_name="bilge_privacy.png",
        mime="image/png"
    )

else:
    st.info("Bir fotoğraf yükleyince otomatik yüz algılama + gizleme başlayacak.")
