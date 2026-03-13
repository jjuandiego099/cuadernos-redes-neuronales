import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------
st.set_page_config(page_title="IA Reconocedor de Números", layout="centered")

# -----------------------------
# TÍTULO Y DESCRIPCIÓN
# -----------------------------
st.title("🧠 Inteligencia Artificial para Identificar Números")

st.subheader("Escribe un número del 0 al 9 y la IA lo clasificará usando un modelo entrenado con Keras y TensorFlow.")

st.write("Dibuja el número en el canvas con fondo negro y lápiz blanco.")

# -----------------------------
# CARGAR MODELO
# -----------------------------
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_entrenado.keras")
    return modelo

modelo = cargar_modelo()

# -----------------------------
# CANVAS PARA DIBUJAR
# -----------------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# BOTÓN PARA PREDECIR
# -----------------------------
if st.button("📸 Capturar y Predecir"):

    if canvas_result.image_data is not None:

        # Convertir imagen a array
        img = canvas_result.image_data

        # Convertir a PIL
        img_pil = Image.fromarray((img[:, :, 0]).astype(np.uint8))

        # Redimensionar a 28x28
        img_resized = img_pil.resize((28, 28))

        # Convertir a numpy
        img_array = np.array(img_resized)

        # Normalizar
        img_array = img_array / 255.0

        # Ajustar forma para el modelo (1,28,28,1)
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predicción
        prediccion = modelo.predict(img_array)
        numero_predicho = np.argmax(prediccion)
        probabilidad = np.max(prediccion)

        # Mostrar resultado
        st.success(f"🔢 El número predicho es: {numero_predicho}")
        st.info(f"📊 Probabilidad: {probabilidad*100:.2f}%")

    else:
        st.warning("Por favor dibuja un número antes de predecir.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("©️ UNAB 2026 - Realizado por Juan Diego Chaparro Garcia")
