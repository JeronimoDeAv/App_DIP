import streamlit as st
from tensorflow.keras.models import load_model
from model_handler import ModelHandler
from image_processor import ImageProcessor
import gdown
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Medical Chest CT Segmentation", page_icon="🩺")

# Google Drive file IDs para cada archivo
unet_scratch_model_id = "1-38XdWMeux6siCzE7NVLqsCX_ZnBAJyF"
unet_transfer_model_id = "1Wf5bzR6Sf2zRfNjFKCmUT6UbgK2MAuP4"
unet_scratch_history_id = "1-56GzsXZ3bpfXGrVOUFQGjOILw_Mc40p"
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

# Inicializar las clases de procesamiento de imágenes
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
        img_array = image_processor.preprocess_image(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Chest CT Image", use_container_width=True)

        # Subir y mostrar la máscara de referencia junto con la imagen y las predicciones
        uploaded_mask_file = st.file_uploader("Upload Ground Truth Mask", type=["png", "jpg", "jpeg"])
        if uploaded_mask_file:
            ground_truth_mask = image_processor.preprocess_image(uploaded_mask_file)
            st.image(ground_truth_mask, caption="Ground Truth Mask", use_container_width=True)

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

            # Comparación visual entre predicciones y ground truth
            if uploaded_mask_file:
                for model_name, prediction in predictions.items():
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(img_array[0])
                    axs[0].set_title("Original Image")
                    axs[1].imshow(ground_truth_mask, cmap="gray")
                    axs[1].set_title("Ground Truth Mask")
                    axs[2].imshow(prediction, cmap="gray")
                    axs[2].set_title(f"Prediction - {model_name}")
                    st.pyplot(fig)

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
                ax.set_xlabel("Épocas")
                ax.set_ylabel("Valor")
                ax.legend()
                st.pyplot(fig)

            # Botón de descarga de la predicción
            for model_name, prediction in predictions.items():
                # Convert the single-channel grayscale prediction to 3-channel RGB format for compatibility
                prediction_rgb = np.repeat(prediction[:, :, np.newaxis], 3, axis=2)
            
                # Save the image in RGB format to avoid "Third dimension must be 3 or 4" error
                buffer = BytesIO()
                plt.imsave(buffer, prediction_rgb, format="jpg")
                buffer.seek(0)
                download_filename = f"{patient_name}_{appointment_date}_{model_name}.jpg"
                st.download_button(
                    label=f"Download {model_name} Prediction",
                    data=buffer,
                    file_name=download_filename,
                    mime="image/jpg"
                )

else:
    st.warning("Please enter doctor and patient information to proceed.")
