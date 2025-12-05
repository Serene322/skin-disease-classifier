# app.py (обновлённый)
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# robust_model_loader.py — вставь этот фрагмент в начало app.py (заменив старую ensure_model)
import os
import gdown
import zipfile
import streamlit as st
from tensorflow import keras

MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)

# Поменяй на ID твоего загруженного ZIP или .h5 в Google Drive
GDRIVE_FILE_ID = "1UNb3jzg1ez9lB5MljHxnFuYsHTAGPgGp"

# локальные имена
ZIP_TARGET = os.path.join(MODEL_DIR, "model_download.zip")   # куда загрузим zip
H5_TARGET = os.path.join(MODEL_DIR, "model.h5")              # если у тебя .h5
EXTRACTED_DIR = os.path.join(MODEL_DIR, "efficientnetv2.keras")  # куда распакуем .keras folder

@st.cache_resource
def ensure_model():
    # Если уже распаковано/есть папка .keras -> загружаем
    if os.path.isdir(EXTRACTED_DIR):
        st.write("Модель уже распакована локально.")
        return keras.models.load_model(EXTRACTED_DIR, compile=False)

    # Если есть один файл .h5 — загрузить и вернуть
    if os.path.exists(H5_TARGET):
        st.write("Найден local .h5 — загружаю...")
        return keras.models.load_model(H5_TARGET, compile=False)

    # Если zip уже скачан — распаковать
    if os.path.exists(ZIP_TARGET):
        st.write(f"ZIP уже скачан: {ZIP_TARGET}, распаковываю...")
        try:
            with zipfile.ZipFile(ZIP_TARGET, 'r') as z:
                z.extractall(MODEL_DIR)
            st.write("Распаковка завершена.")
            # после распаковки ожидаем папку EXTRACTED_DIR или .keras внутри MODEL_DIR
            if os.path.isdir(EXTRACTED_DIR):
                return keras.models.load_model(EXTRACTED_DIR, compile=False)
            # иногда zip может содержать папку с другим именем — ищем .keras папку
            for name in os.listdir(MODEL_DIR):
                p = os.path.join(MODEL_DIR, name)
                if os.path.isdir(p) and name.endswith(".keras"):
                    return keras.models.load_model(p, compile=False)
            raise FileNotFoundError("Не нашёл распакованную .keras папку после unzip.")
        except zipfile.BadZipFile:
            os.remove(ZIP_TARGET)
            st.warning("ZIP файл поврежден — удалён. Попробую скачать заново.")

    # Если ничего нет — скачиваем с Google Drive
    st.info("Скачиваю модель с Google Drive (проверь, что файл публичен и это ZIP или .h5)...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    out = ZIP_TARGET  # попытка скачать zip (или файл .h5, если он там)
    # gdown возвращает путь при успехе, None при провале
    try:
        ret = gdown.download(url, out, quiet=False)
    except Exception as e:
        st.error(f"gdown.download выбросил исключение: {e}")
        raise

    if not ret or not os.path.exists(out):
        # попробуем альтернативный URL с export=download
        alt_url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        st.write("Первичная загрузка не удалась, пробую альтернативный URL...")
        try:
            ret2 = gdown.download(alt_url, out, quiet=False)
        except Exception as e:
            st.error(f"Вторичная загрузка упала: {e}")
            raise
        if not ret2 or not os.path.exists(out):
            raise FileNotFoundError(
                "Не удалось скачать файл с Google Drive. Проверьте права доступа (Anyone with the link) "
                "и то, что вы загрузили не папку. Можно также попробовать загрузить ZIP или .h5."
            )

    # Если скачали — проверим расширение: zip -> распаковать; .h5 -> загрузить сразу
    if zipfile.is_zipfile(out):
        st.write("Файл — zip, распаковываю...")
        with zipfile.ZipFile(out, 'r') as z:
            z.extractall(MODEL_DIR)
        os.remove(out)
        # ищем распакованную .keras папку или любой .keras каталог
        for name in os.listdir(MODEL_DIR):
            p = os.path.join(MODEL_DIR, name)
            if os.path.isdir(p) and name.endswith(".keras"):
                st.write(f"Найдена распакованная модель: {p}")
                return keras.models.load_model(p, compile=False)
        # иначе пробуем найти .h5 внутри
        for root, dirs, files in os.walk(MODEL_DIR):
            for f in files:
                if f.endswith(".h5"):
                    return keras.models.load_model(os.path.join(root, f), compile=False)
        raise FileNotFoundError("После распаковки zip не найдено .keras папки или .h5 файла.")
    else:
        # не zip — возможно скачан единый .h5 или .keras (но если .keras не zip — Keras ожидает zip)
        # переименуем в model.h5 и попытемся загрузить
        st.write("Файл не zip. Проверяю, возможно это .h5...")
        if os.path.exists(out):
            # если имя заканчивается на .h5 — используем его
            if out.endswith(".h5") or out.endswith(".hdf5"):
                return keras.models.load_model(out, compile=False)
            # пробуем загрузить как .keras (keras ожидает .keras быть zip). если не получится — сообщим.
            try:
                return keras.models.load_model(out, compile=False)
            except Exception as e:
                raise RuntimeError(f"Скачан файл, но Keras не может его прочитать: {e}")

    # в общем случае
    raise RuntimeError("Не удалось подготовить модель.")



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
