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
unet_scratch_model_id = "1-afyzXqoQHWDPOy2icnNL__B42dIxDzA"
unet_transfer_model_id = "1-0VQOp3hN5fJQ4WzyhGFFDznbHgov2Li"
unet_scratch_history_id = "1-dWJD1m0C_xXFkXCivrg_pZjNJ7N19yn"
unet_transfer_history_id = "1-717CvxJSww5C-TWKgSW6LHNpjLLh4It"

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

    # Crear metrics_data
    # Crear metrics_data y convertir manualmente cada métrica a array numérico
    metrics_data = {
        'U-Net desde Cero': {
            'loss': np.array(eval(unet_scratch_history['loss'])),
            'val_loss': np.array(eval(unet_scratch_history['val_loss'])),
            'dice_coef': np.array(eval(unet_scratch_history['dice_coef'])),
            'iou_metric': np.array(eval(unet_scratch_history['iou_metric']))
        } if unet_scratch_history is not None else None,
        
        'U-Net Transfer Learning': {
            'loss': np.array(eval(unet_transfer_history['loss'])),
            'val_loss': np.array(eval(unet_transfer_history['val_loss'])),
            'dice_coef': np.array(eval(unet_transfer_history['dice_coef'])),
            'iou_metric': np.array(eval(unet_transfer_history['iou_metric']))
        } if unet_transfer_history is not None else None
    }


    st.write("Contenido de métricas en metrics_data para U-Net Transfer Learning:", metrics_data.get("U-Net Transfer Learning", {}))

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


            if model_choice in ["U-Net desde Cero", "Both"]:
                img_array = image_processor.preprocess_image(uploaded_file, color_mode="grayscale")
                st.write(f"Shape of img_array for scratch model: {img_array.shape}")
                pred = unet_scratch_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net desde Cero", use_container_width=True)
                
                # Añadir predicción y métricas al diccionario
                predictions["U-Net desde Cero"] = processed_mask
                metrics_data["U-Net desde Cero"] = unet_scratch_history
            
            if model_choice in ["U-Net Transfer Learning", "Both"]:
                img_array = image_processor.preprocess_image(uploaded_file, color_mode="rgb")
                st.write(f"Shape of img_array for transfer model: {img_array.shape}")
                pred = unet_transfer_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net Transfer Learning", use_container_width=True)
                
                # Añadir predicción y métricas al diccionario
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

            # Mostrar métricas en Streamlit
            st.subheader("Model Metrics")
            for model_name, metrics in metrics_data.items():
                if metrics is not None:  # Verifica que existan métricas para el modelo
                    st.write(f"**{model_name} Metrics**")
                    
                    # Mostrar los valores finales de cada métrica
                    if 'loss' in metrics:
                        st.write(f"Training Loss: {metrics['loss'][-1]:.4f}")
                    if 'val_loss' in metrics:
                        st.write(f"Validation Loss: {metrics['val_loss'][-1]:.4f}")
                    if 'dice_coef' in metrics:
                        st.write(f"Dice Coefficient: {metrics['dice_coef'][-1]:.4f}")
                    if 'iou_metric' in metrics:
                        st.write(f"IoU Metric: {metrics['iou_metric'][-1]:.4f}")
            
                    # Graficar las métricas disponibles
                    fig, ax = plt.subplots()
                    if 'loss' in metrics:
                        ax.plot(metrics['loss'], label="Training Loss")
                    if 'val_loss' in metrics:
                        ax.plot(metrics['val_loss'], label="Validation Loss")
                    if 'dice_coef' in metrics:
                        ax.plot(metrics['dice_coef'], label="Dice Coefficient")
                    if 'iou_metric' in metrics:
                        ax.plot(metrics['iou_metric'], label="IoU Metric")
                    
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel("Metric Value")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.write(f"No metrics available for {model_name}")

            
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
