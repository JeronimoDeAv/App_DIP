import streamlit as st
from tensorflow.keras.models import load_model
from model_handler import ModelHandler
from image_processor import ImageProcessor
from history_handler import HistoryHandler
import gdown
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from io import BytesIO


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

# Verificar y descargar archivos si no existen
for file_id, output_path, name in [
    (unet_scratch_model_id, unet_scratch_model_path, "modelo U-Net desde cero"),
    (unet_transfer_model_id, unet_transfer_model_path, "modelo U-Net Transfer Learning"),
    (unet_scratch_history_id, unet_scratch_history_path, "historial U-Net desde cero"),
    (unet_transfer_history_id, unet_transfer_history_path, "historial U-Net Transfer Learning")
]:
    if not os.path.exists(output_path):
        st.write(f"Descargando {name}...")
        download_file_from_drive(file_id, output_path)

# Cargar los modelos y archivos de historial
try:
    unet_scratch_model = load_model(unet_scratch_model_path)
    unet_transfer_model = load_model(unet_transfer_model_path)
    unet_scratch_history = np.load(unet_scratch_history_path, allow_pickle=True) if os.path.exists(unet_scratch_history_path) else None
    unet_transfer_history = np.load(unet_transfer_history_path, allow_pickle=True) if os.path.exists(unet_transfer_history_path) else None
    st.success("¡Modelos e historiales cargados exitosamente!")
except Exception as e:
    st.error(f"Error al cargar los modelos o historiales: {e}")


image_processor = ImageProcessor()

# Inicializar la app
st.title("Medical Chest CT Segmentation")
st.subheader("Doctor and Patient Information")

# Información del doctor y paciente
doctor_id = st.text_input("Doctor ID")
patient_name = st.text_input("Patient Name")
appointment_date = st.date_input("Date", datetime.date.today())

if doctor_id and patient_name:
    st.success("Information entered successfully. Proceed to image upload.")

    # Cargar imagen y seleccionar modelo
    st.subheader("Step 1: Upload Chest CT Image")
    uploaded_file = st.file_uploader("Upload a Chest CT Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Procesar la imagen para hacerla compatible con el modelo
        img_array = image_processor.preprocess_image(uploaded_file)  # Define `img_array` aquí
        st.image(uploaded_file, caption="Uploaded Chest CT Image", use_container_width=True)

        st.subheader("Step 2: Select Model(s) for Prediction")
        model_choice = st.radio("Choose a model:", ["U-Net desde Cero", "U-Net Transfer Learning", "Both"])

        if st.button("Generate Prediction"):
            predictions = {}
            metrics_data = {}

            # Ejecutar el modelo seleccionado
            if model_choice in ["U-Net desde Cero", "Both"] and unet_scratch_history is not None:
                pred = unet_scratch_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net desde Cero", use_container_width=True)
                predictions["U-Net desde Cero"] = processed_mask
                metrics_data["U-Net desde Cero"] = unet_scratch_history

            if model_choice in ["U-Net Transfer Learning", "Both"] and unet_transfer_history is not None:
                pred = unet_transfer_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net Transfer Learning", use_container_width=True)
                predictions["U-Net Transfer Learning"] = processed_mask
                metrics_data["U-Net Transfer Learning"] = unet_transfer_history

            # Mostrar métricas
            st.subheader("Model Metrics")
            for model_name, metrics in metrics_data.items():
                st.write(f"**{model_name} Metrics**")
                st.write(f"IoU: {metrics['iou_metric'][-1]:.4f}")
                st.write(f"Dice Coefficient: {metrics['dice_coef'][-1]:.4f}")
                st.write(f"Validation Loss: {metrics['val_loss'][-1]:.4f}")

                # Graficar métricas
                fig, ax = plt.subplots()
                ax.plot(metrics['loss'], label="Training Loss")
                ax.plot(metrics['val_loss'], label="Validation Loss")
                ax.plot(metrics['dice_coef'], label="Dice Coefficient")
                ax.plot(metrics['iou_metric'], label="IoU Metric")
                ax.legend()
                st.pyplot(fig)

            # Opción de exportar
            if st.button("Export Prediction and Metrics"):
                # Preparar el contenido del informe
                report = f"Doctor ID: {doctor_id}\nPatient Name: {patient_name}\nAppointment Date: {appointment_date}\n\nPredictions and Metrics:\n"
                for model_name, metrics in metrics_data.items():
                    report += f"\nModel: {model_name}\nIoU: {metrics['iou_metric'][-1]:.4f}\nDice Coefficient: {metrics['dice_coef'][-1]:.4f}\nValidation Loss: {metrics['val_loss'][-1]:.4f}\n"
            
                # Convertir el informe en un flujo de bytes
                report_bytes = BytesIO(report.encode('utf-8'))
            
                # Botón para descargar el informe
                st.download_button(
                    label="Download Report",
                    data=report_bytes,
                    file_name=f"{patient_name}_prediction_report.txt",
                    mime="text/plain"
                )
else:
    st.warning("Please enter doctor and patient information to proceed.")
