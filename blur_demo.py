import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Bilge Face Blur", page_icon="📷", layout="centered")

st.title("📷 Bilge Face Blur Demo")
st.write("Bir fotoğraf yükle → yüz(leri) otomatik bulsun → sadece yüzleri blur’lasın.")

# --- Ayarlar ---
blur_strength = st.slider("Blur seviyesi", 1, 50, 15)  # 1..50
expand = st.slider("Yüz kutusunu büyüt (px)", 0, 80, 20)

uploaded = st.file_uploader("Fotoğraf yükle (jpg/png)", type=["jpg", "jpeg", "png"])

def ensure_odd(k: int) -> int:
    # GaussianBlur için kernel tek sayı olmalı
    return k if k % 2 == 1 else k + 1

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(pil_img)

    # OpenCV BGR formatına çevir
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Haar cascade (OpenCV içinde hazır gelir)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Yüzleri bul
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    output = img_bgr.copy()

    if len(faces) == 0:
        st.warning("Yüz bulunamadı. Daha net bir fotoğraf deneyebilir misin?")
    else:
        for (x, y, w, h) in faces:
            # Kutuyu biraz büyüt (daha iyi gizleme)
            x1 = clamp(x - expand, 0, output.shape[1] - 1)
            y1 = clamp(y - expand, 0, output.shape[0] - 1)
            x2 = clamp(x + w + expand, 0, output.shape[1])
            y2 = clamp(y + h + expand, 0, output.shape[0])

            face_roi = output[y1:y2, x1:x2]

            k = ensure_odd(blur_strength)
            blurred_face = cv2.GaussianBlur(face_roi, (k, k), 0)

            output[y1:y2, x1:x2] = blurred_face

        st.success(f"Bulunan yüz sayısı: {len(faces)}")

    # Sonucu RGB'ye geri çevir
    out_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Orijinal")
        st.image(img_rgb, use_container_width=True)
    with c2:
        st.subheader("Yüzler Blur")
        st.image(out_rgb, use_container_width=True)

    # İndirme
    out_pil = Image.fromarray(out_rgb)
    import io
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button(
        "⬇️ Blur’lu görseli indir (PNG)",
        data=buf.getvalue(),
        file_name="bilge_face_blur.png",
        mime="image/png"
    )
