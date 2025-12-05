# app.py (обновлённый)
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# -------- robust loader (вставь вместо старой ensure_model) -----------
import os, io, stat, time
import gdown
import zipfile
import streamlit as st
from tensorflow import keras

MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)

# Твой новый ZIP FILE_ID (проверь, что это ZIP-файл в Drive)
GDRIVE_FILE_ID = "1H_oLka9VEo6a0A1Kvi56Ye7k5xqnjneC"  # замените на ваш ID (пример)
ZIP_TARGET = os.path.join(MODEL_DIR, "efficientnetv2.keras.zip")
EXTRACTED_DIR = os.path.join(MODEL_DIR, "efficientnetv2.keras")

def _debug_dump_head(path, n=512):
    try:
        with open(path, "rb") as f:
            head = f.read(n)
        # Покажем первые байты (в основном для html ошибок Google Drive)
        try:
            s = head.decode("utf-8", errors="replace")
        except:
            s = repr(head)
        return s
    except Exception as e:
        return f"<cannot read file head: {e}>"

@st.cache_resource
def ensure_model():
    st.write("ensure_model: START")
    st.write(f"MODEL_DIR={MODEL_DIR}, ZIP_TARGET={ZIP_TARGET}, EXTRACTED_DIR={EXTRACTED_DIR}")

    # 1) если уже распаковано
    if os.path.isdir(EXTRACTED_DIR):
        st.write("ensure_model: found already-extracted folder:", EXTRACTED_DIR)
        return keras.models.load_model(EXTRACTED_DIR, compile=False)

    # 2) если ZIP существует локально, проверим его
    if os.path.exists(ZIP_TARGET):
        st.write("ensure_model: ZIP_TARGET exists, size (bytes):", os.path.getsize(ZIP_TARGET))
        # если размер маленький — покажем первые байты для диагностики
        if os.path.getsize(ZIP_TARGET) < 1000:
            st.warning("ensure_model: downloaded file is very small (<1000 B) — likely not a zip.")
            st.text(_debug_dump_head(ZIP_TARGET, n=1024))
        if zipfile.is_zipfile(ZIP_TARGET):
            st.write("ensure_model: ZIP seems valid, extracting...")
            with zipfile.ZipFile(ZIP_TARGET, 'r') as z:
                z.extractall(MODEL_DIR)
            st.write("ensure_model: extraction done. Listing", os.listdir(MODEL_DIR))
            # найти папку .keras или .h5
            for name in os.listdir(MODEL_DIR):
                p = os.path.join(MODEL_DIR, name)
                st.write(" - found:", name)
                if os.path.isdir(p) and name.endswith(".keras"):
                    st.write("ensure_model: loading .keras folder:", p)
                    return keras.models.load_model(p, compile=False)
                if os.path.isfile(p) and p.endswith(".h5"):
                    st.write("ensure_model: loading .h5 file:", p)
                    return keras.models.load_model(p, compile=False)
            # если не нашли — пробуем рекурсивно найти .keras/.h5
            for root, dirs, files in os.walk(MODEL_DIR):
                for d in dirs:
                    if d.endswith(".keras"):
                        p = os.path.join(root, d)
                        st.write("ensure_model: found nested .keras:", p)
                        return keras.models.load_model(p, compile=False)
                for f in files:
                    if f.endswith(".h5"):
                        p = os.path.join(root, f)
                        st.write("ensure_model: found nested .h5:", p)
                        return keras.models.load_model(p, compile=False)
            raise FileNotFoundError("After extraction no .keras folder or .h5 file found in model_files.")
        else:
            st.warning("ensure_model: ZIP_TARGET exists but is not a valid zip file. Dumping head:")
            st.text(_debug_dump_head(ZIP_TARGET, n=2048))
            # удалим подозрительный файл, чтобы попытаться скачать заново
            try:
                os.remove(ZIP_TARGET)
                st.write("ensure_model: removed invalid zip to retry download.")
            except Exception as e:
                st.write("ensure_model: cannot remove invalid zip:", e)

    # 3) Скачиваем ZIP (попробуем два варианта URL)
    url1 = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    url2 = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    st.write("ensure_model: downloading from url1:", url1)
    try:
        gdown.download(url1, ZIP_TARGET, quiet=False)
    except Exception as e:
        st.warning("ensure_model: gdown.download(url1) failed: " + str(e))

    if not os.path.exists(ZIP_TARGET) or os.path.getsize(ZIP_TARGET) < 100:
        st.write("ensure_model: url1 failed or small file, trying url2...")
        try:
            gdown.download(url2, ZIP_TARGET, quiet=False)
        except Exception as e:
            st.warning("ensure_model: gdown.download(url2) failed: " + str(e))

    # Проверка результата
    if not os.path.exists(ZIP_TARGET):
        raise FileNotFoundError(
            "Download failed: ZIP not found. Check Google Drive file permissions (set 'Anyone with the link' and make sure it's a FILE, not a folder)."
        )

    st.write("ensure_model: downloaded size (bytes):", os.path.getsize(ZIP_TARGET))
    if not zipfile.is_zipfile(ZIP_TARGET):
        st.warning("Downloaded file is NOT a zip file. First bytes (for debug):")
        st.text(_debug_dump_head(ZIP_TARGET, n=2048))
        raise RuntimeError("Downloaded file is not a zip. Likely Google Drive returned an HTML page (access denied / quota).")

    # распаковать и загрузить (повторяем ту же логику)
    with zipfile.ZipFile(ZIP_TARGET, 'r') as z:
        z.extractall(MODEL_DIR)
    st.write("ensure_model: extraction complete. Listing:", os.listdir(MODEL_DIR))
    for name in os.listdir(MODEL_DIR):
        p = os.path.join(MODEL_DIR, name)
        st.write(" - found after extract:", name)
        if os.path.isdir(p) and name.endswith(".keras"):
            st.write("ensure_model: loading .keras from", p)
            return keras.models.load_model(p, compile=False)
        if os.path.isfile(p) and p.endswith(".h5"):
            st.write("ensure_model: loading .h5 from", p)
            return keras.models.load_model(p, compile=False)

    # рекурсивный поиск
    for root, dirs, files in os.walk(MODEL_DIR):
        for d in dirs:
            if d.endswith(".keras"):
                p = os.path.join(root, d)
                st.write("ensure_model: found nested .keras:", p)
                return keras.models.load_model(p, compile=False)
        for f in files:
            if f.endswith(".h5"):
                p = os.path.join(root, f)
                st.write("ensure_model: found nested .h5:", p)
                return keras.models.load_model(p, compile=False)

    raise FileNotFoundError("Extraction finished but .keras folder/.h5 file not found.")
# -------- end robust loader -----------



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
