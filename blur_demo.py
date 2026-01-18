import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Bilge Privacy Lab", page_icon="🛡️", layout="wide")

st.title("🛡️ Bilge Privacy Lab")
st.caption("Fotoğraf yükle → yüzleri otomatik bul → gizle → indir (Streamlit Cloud uyumlu)")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def get_face_cascade():
    # OpenCV'nin kendi paket içi haarcascade yolu (ek dosya gerektirmez)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    c = cv2.CascadeClassifier(cascade_path)
    return c

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def resize_if_needed(bgr: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def blur_roi(bgr: np.ndarray, x: int, y: int, w: int, h: int, k: int) -> None:
    roi = bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return
    k = max(3, k)
    if k % 2 == 0:
        k += 1
    bgr[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)

def pixelate_roi(bgr: np.ndarray, x: int, y: int, w: int, h: int, block: int) -> None:
    roi = bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return
    block = max(4, int(block))
    small_w = max(1, w // block)
    small_h = max(1, h // block)
    temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    bgr[y:y+h, x:x+w] = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def detect_faces(gray: np.ndarray, scaleFactor: float, minNeighbors: int, minSize: int):
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(minSize, minSize),
    )
    return faces

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Kontroller")

    mode = st.selectbox("Gizleme tipi", ["Gaussian Blur", "Pixelate"], index=0)

    blur_k = st.slider("Blur gücü (tek sayı)", 3, 99, 31, step=2)
    pixel_block = st.slider("Piksel blok boyutu", 4, 50, 14)

    st.divider()
    st.subheader("Yüz algılama")
    min_face = st.slider("Min yüz boyutu", 20, 200, 40)
    scaleFactor = st.slider("scaleFactor (hassasiyet)", 1.05, 1.50, 1.20, step=0.01)
    minNeighbors = st.slider("minNeighbors (filtre)", 1, 10, 4)

    show_boxes = st.toggle("Yüz kutularını göster", value=True)
    background_blur = st.toggle("Arka plan blur (yüz net değil, yüz gizli)", value=False)
    bg_blur_k = st.slider("Arka plan blur gücü", 3, 99, 21, step=2)

st.write("### Fotoğraf yükle")
uploaded = st.file_uploader("JPG/PNG yükle", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Bir görsel yükleyince yüz algılama başlayacak.")
    st.stop()

# Büyük görseller Streamlit Cloud’da bazen 1 dk sonra düşürür → önce küçültüyoruz
pil_img = Image.open(uploaded)
bgr = pil_to_bgr(pil_img)
bgr = resize_if_needed(bgr, max_side=1400)

gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

faces = detect_faces(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=min_face)

processed = bgr.copy()

# Arka plan blur (isteğe bağlı)
if background_blur:
    k = bg_blur_k
    if k % 2 == 0:
        k += 1
    processed = cv2.GaussianBlur(processed, (k, k), 0)
    # sonra yüz bölgelerini orijinalden alıp ÜSTÜNE gizleme uygulayacağız (yani arka plan blur + yüz gizli)
    # Burada zaten processed blur'lu; yüz ROI'sini tekrar işleyeceğiz.

# Yüzleri gizle
for (x, y, w, h) in faces:
    pad = int(0.15 * w)
    x2 = max(0, x - pad)
    y2 = max(0, y - pad)
    w2 = min(processed.shape[1] - x2, w + 2 * pad)
    h2 = min(processed.shape[0] - y2, h + 2 * pad)

    if mode == "Gaussian Blur":
        blur_roi(processed, x2, y2, w2, h2, blur_k)
    else:
        pixelate_roi(processed, x2, y2, w2, h2, pixel_block)

    if show_boxes:
        cv2.rectangle(processed, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

col1, col2 = st.columns(2)

with col1:
    st.write("### Orijinal")
    st.image(bgr_to_pil(bgr), use_container_width=True)

with col2:
    st.write(f"### Gizlenmiş (Bulunan yüz: {len(faces)})")
    out_pil = bgr_to_pil(processed)
    st.image(out_pil, use_container_width=True)

# İndir
out_bytes = np.array(out_pil.convert("RGB"))
# PNG indir (kalite kaybı yok)
_, png = cv2.imencode(".png", cv2.cvtColor(out_bytes, cv2.COLOR_RGB2BGR))
st.download_button(
    "⬇️ PNG olarak indir",
    data=png.tobytes(),
    file_name="bilge_privacy.png",
    mime="image/png",
)
