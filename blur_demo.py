import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="Bilge Privacy Lab", page_icon="🛡️", layout="centered")
st.title("🛡️ Bilge Privacy Lab")
st.caption("Tek dosya, modüler. Yüz algıla → gizle → indir. (Cloud üzerinde çalışır)")

# ----------------------------
# HELPERS
# ----------------------------
def ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    rgb = np.array(img_pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pixelate(roi_bgr: np.ndarray, pixel_size: int = 12) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    pixel_size = max(2, int(pixel_size))
    small = cv2.resize(roi_bgr, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_face_effect(output_bgr: np.ndarray, box, mode: str, strength: int, pixel_size: int):
    x1, y1, x2, y2 = box
    roi = output_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return

    if mode == "Gaussian":
        k = ensure_odd(strength)
        output_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    elif mode == "Pixelate":
        output_bgr[y1:y2, x1:x2] = pixelate(roi, pixel_size=pixel_size)
    elif mode == "Black bar":
        output_bgr[y1:y2, x1:x2] = 0

def draw_boxes(img_bgr: np.ndarray, boxes, thickness=2):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)

# ----------------------------
# SIDEBAR (FEATURE FLAGS)
# ----------------------------
st.sidebar.header("⚙️ Kontroller")

privacy_mode = st.sidebar.toggle("🧷 Privacy Mode (tek tık preset)", value=True)

# Varsayılanlar
default_face_mode = "Gaussian"
default_strength = 31
default_pixel = 14
default_expand = 20
default_min_size = 40
default_show_boxes = False
default_background_blur = False
default_bg_strength = 21

# Privacy mode preset
if privacy_mode:
    face_mode = st.sidebar.selectbox("Yüz gizleme tipi", ["Gaussian", "Pixelate", "Black bar"], index=1)
    strength = st.sidebar.slider("Gaussian gücü", 1, 51, default_strength, step=2)
    pixel_size = st.sidebar.slider("Pixel boyutu", 4, 40, default_pixel, step=1)
    expand = st.sidebar.slider("Kutu büyüt (px)", 0, 120, 30)
    min_size = st.sidebar.slider("Min yüz boyutu", 20, 120, 40)
    show_boxes = st.sidebar.toggle("Yüz kutularını göster", value=False)
    background_blur = st.sidebar.toggle("Arka plan blur (yüz net)", value=False)
    bg_strength = st.sidebar.slider("Arka plan blur gücü", 1, 51, 21, step=2)
else:
    st.sidebar.subheader("Modüller")
    face_mode = st.sidebar.selectbox("Yüz gizleme tipi", ["Gaussian", "Pixelate", "Black bar"], index=0)
    strength = st.sidebar.slider("Gaussian gücü", 1, 51, 15, step=2)
    pixel_size = st.sidebar.slider("Pixel boyutu", 4, 40, 12, step=1)
    expand = st.sidebar.slider("Kutu büyüt (px)", 0, 120, 20)
    min_size = st.sidebar.slider("Min yüz boyutu", 20, 120, 40)
    show_boxes = st.sidebar.toggle("Yüz kutularını göster", value=False)
    background_blur = st.sidebar.toggle("Arka plan blur (yüz net)", value=False)
    bg_strength = st.sidebar.slider("Arka plan blur gücü", 1, 51, 21, step=2)

uploaded = st.file_uploader("📤 Fotoğraf yükle (jpg/png)", type=["jpg", "jpeg", "png"])

# ----------------------------
# FACE DETECTION + PROCESSING
# ----------------------------
if not uploaded:
    st.info("Yukarıdan bir fotoğraf yükle. Sonuçlar burada görünecek.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_bgr = pil_to_bgr(pil_img)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(min_size, min_size),
)

# Yüz kutularını "x1,y1,x2,y2" formatına çevir ve büyüt
h, w = img_bgr.shape[:2]
boxes = []
for (x, y, fw, fh) in faces:
    x1 = clamp(x - expand, 0, w - 1)
    y1 = clamp(y - expand, 0, h - 1)
    x2 = clamp(x + fw + expand, 0, w)
    y2 = clamp(y + fh + expand, 0, h)
    boxes.append((x1, y1, x2, y2))

# Arka plan blur (yüzler net kalsın) istenirse:
output_bgr = img_bgr.copy()
if background_blur:
    kbg = ensure_odd(bg_strength)
    blurred_all = cv2.GaussianBlur(output_bgr, (kbg, kbg), 0)

    mask = np.zeros
