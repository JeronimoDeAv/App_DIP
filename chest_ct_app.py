import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model_handler import ModelHandler
from image_processor import ImageProcessor


class ChestCTApp:
    def __init__(self, model_handler, image_processor):
        self.model_handler = model_handler
        self.image_processor = image_processor

    def load_history(self, model_name):
        try:
            history_data = np.load(f"{model_name}_history.npz")
            return history_data
        except FileNotFoundError:
            st.error("Historial de entrenamiento no encontrado para este modelo.")
            return None

    def plot_history(self, history_data, model_name):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(history_data['loss'], label='Entrenamiento')
        ax1.plot(history_data['val_loss'], label='Validación')
        ax1.set_title(f'Pérdida - {model_name}')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Pérdida')
        ax1.legend()

        ax2.plot(history_data['dice_coef'], label='Entrenamiento')
        ax2.plot(history_data['val_dice_coef'], label='Validación')
        ax2.set_title(f'Dice Coeficiente - {model_name}')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Dice Coeficiente')
        ax2.legend()

        st.pyplot(fig)

    def run(self):
        st.title("Aplicación de Segmentación para CT de Tórax")
        st.write("Sube una imagen de CT y selecciona un modelo para ver la predicción y métricas.")

        uploaded_file = st.file_uploader("Sube una imagen de CT de tórax", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen cargada", use_column_width=True)

            model_name = st.selectbox("Selecciona un modelo", list(self.model_handler.models.keys()))
            image_array = self.image_processor.preprocess_image(image)

            history_data = self.load_history(model_name)
            if history_data is not None:
                st.subheader("Historial de Entrenamiento")
                self.plot_history(history_data, model_name)
                st.write("**Métricas finales de entrenamiento**")
                st.write(f"Última pérdida (entrenamiento): {history_data['loss'][-1]:.4f}")
                st.write(f"Última pérdida (validación): {history_data['val_loss'][-1]:.4f}")
                st.write(f"Último Dice Coeficiente (entrenamiento): {history_data['dice_coef'][-1]:.4f}")
                st.write(f"Último Dice Coeficiente (validación): {history_data['val_dice_coef'][-1]:.4f}")

            if st.button("Realizar predicción"):
                prediction = self.model_handler.predict(model_name, image_array)
                prediction_post = self.image_processor.postprocess_prediction(prediction)

                st.subheader("Resultados de Predicción")
                st.image(prediction, caption="Predicción Original", use_column_width=True)
                st.image(prediction_post, caption="Predicción con Post-Procesamiento", use_column_width=True)
