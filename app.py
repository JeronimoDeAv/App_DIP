import streamlit as st
from model_handler import ModelHandler
from image_processor import ImageProcessor
from history_handler import HistoryHandler
import datetime
import matplotlib.pyplot as plt
import gdown
import os
from tensorflow.keras.models import load_model
import numpy as np

st.set_page_config(page_title="Medical Chest CT Segmentation", page_icon="🩺")

# Google Drive file IDs para cada archivo
unet_scratch_model_id = "1F8zkCMlT2eBRjJ5gjhxzp-yq_7zFkr-h"
unet_transfer_model_id = "1Wf5bzR6Sf2zRfNjFKCmUT6UbgK2MAuP4"
unet_scratch_history_id = "1SiOtLlKK2GsZ9VsIW_LHlH-VEcFBrsWk"
unet_transfer_history_id = "16mklVOSDXywiPx7z1RqVACjJMzn29kni"

# Rutas locales para guardar los archivos
unet_scratch_model_path = "unet_scratch.keras"
unet_transfer_model_path = "unet_transfer.keras"
unet_scratch_history_path = "unet_scratch_history.npz"
unet_transfer_history_path = "unet_transfer_history.npz"

# Función para descargar un archivo de Google Drive usando gdown
def download_file_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Verifica si los archivos ya existen y descárgalos si es necesario
if not os.path.exists(unet_scratch_model_path):
    st.write("Descargando modelo U-Net desde cero...")
    download_file_from_drive(unet_scratch_model_id, unet_scratch_model_path)

if not os.path.exists(unet_transfer_model_path):
    st.write("Descargando modelo U-Net Transfer Learning...")
    download_file_from_drive(unet_transfer_model_id, unet_transfer_model_path)

if not os.path.exists(unet_scratch_history_path):
    st.write("Descargando historial U-Net desde cero...")
    download_file_from_drive(unet_scratch_history_id, unet_scratch_history_path)

if not os.path.exists(unet_transfer_history_path):
    st.write("Descargando historial U-Net Transfer Learning...")
    download_file_from_drive(unet_transfer_history_id, unet_transfer_history_path)

# Cargar los modelos y archivos de historial
st.write("Cargando modelos y historiales...")
unet_scratch_model = load_model(unet_scratch_model_path)
unet_transfer_model = load_model(unet_transfer_model_path)
unet_scratch_history = np.load(unet_scratch_history_path, allow_pickle=True)
unet_transfer_history = np.load(unet_transfer_history_path, allow_pickle=True)

st.success("¡Modelos e historiales cargados exitosamente!")

# Inicializar componentes de la app de Streamlit
st.title("Medical Chest CT Segmentation")
st.subheader("Doctor and Patient Information")

# Recolectar información del doctor y paciente
doctor_id = st.text_input("Doctor ID")
patient_name = st.text_input("Patient Name")
appointment_date = st.date_input("Date", datetime.date.today())

if doctor_id and patient_name:
    st.success("Información ingresada correctamente. Procede a cargar la imagen.")

    # Carga de imagen y selección de modelo
    st.subheader("Paso 1: Subir Imagen de TC de Tórax")
    uploaded_file = st.file_uploader("Sube una imagen de TC de tórax", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image_processor = ImageProcessor()  # Instancia de ImageProcessor
        img_array = image_processor.preprocess_image(uploaded_file)
        st.image(uploaded_file, caption="Imagen de TC de Tórax Cargada", use_container_width=True)

        st.subheader("Paso 2: Selecciona el Modelo para Predicción")
        model_choice = st.radio("Elige un modelo:", ["U-Net desde Cero", "U-Net Transfer Learning", "Ambos"])

        if st.button("Generar Predicción"):
            predictions = {}
            metrics_data = {}

            # Ejecuta los modelos seleccionados
            if model_choice in ["U-Net desde Cero", "Ambos"]:
                pred = unet_scratch_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Predicción - U-Net desde Cero", use_container_width=True)
                predictions["U-Net desde Cero"] = processed_mask
                metrics_data["U-Net desde Cero"] = unet_scratch_history

            if model_choice in ["U-Net Transfer Learning", "Ambos"]:
                pred = unet_transfer_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Predicción - U-Net Transfer Learning", use_container_width=True)
                predictions["U-Net Transfer Learning"] = processed_mask
                metrics_data["U-Net Transfer Learning"] = unet_transfer_history

            # Mostrar métricas
            st.subheader("Métricas del Modelo")
            for model_name, metrics in metrics_data.items():
                st.write(f"**Métricas de {model_name}**")
                st.write(f"IoU: {metrics['iou_metric'][-1]:.4f}")
                st.write(f"Dice Coefficient: {metrics['dice_coef'][-1]:.4f}")
                st.write(f"Validación Pérdida: {metrics['val_loss'][-1]:.4f}")

                # Gráfica de métricas
                fig, ax = plt.subplots()
                ax.plot(metrics['loss'], label="Pérdida Entrenamiento")
                ax.plot(metrics['val_loss'], label="Pérdida Validación")
                ax.plot(metrics['dice_coef'], label="Coeficiente Dice")
                ax.plot(metrics['iou_metric'], label="Métrica IoU")
                ax.legend()
                st.pyplot(fig)

            # Opción para exportar predicción y métricas
            if st.button("Exportar Predicción y Métricas"):
                # Preparar contenido del informe
                report = f"""
                Doctor ID: {doctor_id}
                Nombre del Paciente: {patient_name}
                Fecha de Cita: {appointment_date}

                Predicciones y Métricas:
                """
                for model_name, metrics in metrics_data.items():
                    report += f"\nModelo: {model_name}\n"
                    report += f"IoU: {metrics['iou_metric'][-1]:.4f}\n"
                    report += f"Coeficiente Dice: {metrics['dice_coef'][-1]:.4f}\n"
                    report += f"Pérdida Validación: {metrics['val_loss'][-1]:.4f}\n"

                # Botón de descarga para el informe en texto
                st.download_button(
                    label="Descargar Informe",
                    data=report,
                    file_name=f"{patient_name}_prediction_report.txt",
                    mime="text/plain"
                )
else:
    st.warning("Por favor ingresa la información del doctor y paciente para continuar.")
