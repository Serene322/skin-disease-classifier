# app.py (обновлённый)
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import os
import gdown
from tensorflow import keras
import streamlit as st

# Папка для хранения модели
MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive FILE_ID вашей модели
GDRIVE_FILE_ID = "1UNb3jzg1ez9lB5MljHxnFuYsHTAGPgGp"

# Путь к локальному файлу модели
MODEL_PATH = os.path.join(MODEL_DIR, "efficientnetv2.keras")

# Кэширование модели, чтобы не скачивать при каждом cold start
@st.cache_resource
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        st.info("Скачиваю модель с Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

# Загружаем модель
model = ensure_model()
st.success("Модель загружена!")


# ========== ПАРАМЕТРЫ ==========
IMG_SIZE = (224, 224)   # как у тебя
MODEL_PATH = "efficientnetv2.keras"   # или "efficientnetv2_final.h5"
IMAGE_DISPLAY_WIDTH = 400  # ширина предпросмотра в пикселях (подстрой под телефон/ПК)
# ==============================

st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("Классификатор кожных заболеваний")
st.write("Загрузите фото — модель выдаст топ-3 вероятных диагноза с процентами.")

# === Классы и перевод на русский ===
CLASS_NAMES = [
    "Melanocytic Nevi (NV)",
    "Basal Cell Carcinoma (BCC)",
    "Melanoma",
    "Eczema",
    "Warts, Molluscum and other Viral Infections",
    "Atopic Dermatitis",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis, Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Tinea, Ringworm, Candidiasis and other Fungal Infections"
]

CLASS_NAMES_RU = [
    "Меланоцитарные невусы (NV)",
    "Базальноклеточная карцинома (BCC)",
    "Меланома",
    "Экзема",
    "Бородавки, моллюск и другие вирусные инфекции",
    "Атопический дерматит",
    "Доброкачественные кератозоподобные образования (BKL)",
    "Псориаз, лишай плоский и родственные заболевания",
    "Себорейные кератозы и другие доброкачественные новообразования",
    "Грибковые инфекции (тиния, кандидоз и др.)"
]

# === Загрузка модели (лениво, при первом запросе) ===
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    # Загружаем в режиме infer (compile=False) — безопаснее для кастомного loss/schedule
    model = keras.models.load_model(path, compile=False)
    return model

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Не удалось загрузить модель по пути '{MODEL_PATH}': {e}")
    st.stop()

# === Определяем подходящий ресемплер для Pillow (совместимо с разными версиями) ===
try:
    resample_method = Image.Resampling.LANCZOS  # Pillow >= 9.1 / 10+
except Exception:
    # fallback для старых версий Pillow
    try:
        resample_method = Image.LANCZOS
    except Exception:
        resample_method = Image.BICUBIC  # гарантированный запасной вариант

# === Вспомогательные функции ===
def prepare_image(pil_image: Image.Image, img_size=IMG_SIZE):
    # Привести к RGB
    img = pil_image.convert("RGB")
    # Менее агрессивный ресайз — используем выбранный resample_method
    img = img.resize(img_size, resample=resample_method)
    arr = np.asarray(img).astype(np.float32)
    # preprocess_input для EfficientNetV2 (делает нужную нормализацию)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    return arr

def predict_top3(model, prepped_image):
    preds = model.predict(prepped_image)  # (1, num_classes)
    probs = preds[0]
    # защита от числовых артефактов: нормируем
    if np.sum(probs) <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / np.sum(probs)
    idx = np.argsort(-probs)[:3]
    return [(int(i), float(probs[i])) for i in idx]

# === UI: загрузка файла ===
uploaded_file = st.file_uploader("Загрузить фото", type=["jpg","jpeg","png"], accept_multiple_files=False)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Невозможно открыть изображение: {e}")
        st.stop()

    # Используем width вместо устаревшего use_column_width
    st.image(image, caption="Загруженное изображение", width=IMAGE_DISPLAY_WIDTH)

    if st.button("Прогнать через модель"):
        with st.spinner("Загрузка... Подождите, идёт предсказание"):
            try:
                x = prepare_image(image, IMG_SIZE)
                top3 = predict_top3(model, x)
            except Exception as e:
                st.error(f"Ошибка при подготовке/предсказании: {e}")
                raise

        st.success("Готово — топ-3 предположения:")
        for rank, (class_idx, prob) in enumerate(top3, start=1):
            name_en = CLASS_NAMES[class_idx]
            name_ru = CLASS_NAMES_RU[class_idx]
            st.markdown(f"**{rank}. {name_ru}** ({name_en}) — **{prob*100:.1f}%**")

        st.markdown("---")
        st.info("Если хотите проверить другое фото — загрузите новый файл выше.")
else:
    st.write("Нажмите кнопку «Загрузить фото», чтобы выбрать изображение из галереи.")
