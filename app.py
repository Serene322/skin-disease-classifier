# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ========== Настройки ==========
MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)

REPO_ID = "SereneYiver/skin-disease-classifier"  # твой репозиторий на HF
FILENAME = "efficientnetv2.keras"  # прямо .keras, НЕ ZIP
HF_TOKEN = "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"

IMG_SIZE = (224, 224)
IMAGE_DISPLAY_WIDTH = 400

# Классы и перевод
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

# ==============================

st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("Классификатор кожных заболеваний")
st.write("Загрузите фото — модель выдаст топ‑3 вероятных диагноза с процентами.")

# === Загрузка модели ===
@st.cache_resource
def ensure_model():
    # Скачиваем .keras напрямую
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=HF_TOKEN,
        cache_dir=MODEL_DIR
    )
    st.info(f"Модель загружена из {model_path}")
    return keras.models.load_model(model_path, compile=False)

try:
    model = ensure_model()
    st.success("Модель загружена!")
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")
    st.stop()

# === Ресемплер для Pillow ===
try:
    resample_method = Image.Resampling.LANCZOS
except Exception:
    try:
        resample_method = Image.LANCZOS
    except Exception:
        resample_method = Image.BICUBIC

# === Вспомогательные функции ===
def prepare_image(pil_image: Image.Image, img_size=IMG_SIZE):
    img = pil_image.convert("RGB")
    img = img.resize(img_size, resample=resample_method)
    arr = np.asarray(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_top3(model, prepped_image):
    preds = model.predict(prepped_image)
    probs = preds[0]
    probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs)/len(probs)
    idx = np.argsort(-probs)[:3]
    return [(int(i), float(probs[i])) for i in idx]

# === UI: загрузка фото ===
uploaded_file = st.file_uploader("Загрузить фото", type=["jpg","jpeg","png"], accept_multiple_files=False)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Невозможно открыть изображение: {e}")
        st.stop()

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

