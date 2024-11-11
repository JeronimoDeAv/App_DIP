import streamlit as st
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





# Load models and history files
unet_scratch_model = load_model(unet_scratch_model_path)
unet_transfer_model = load_model(unet_transfer_model_path)
unet_scratch_history = np.load(unet_scratch_history_path, allow_pickle=True)
unet_transfer_history = np.load(unet_transfer_history_path, allow_pickle=True)

# Initialize Streamlit app components
st.title("Medical Chest CT Segmentation")
st.subheader("Doctor and Patient Information")

# Collect doctor and patient info
doctor_id = st.text_input("Doctor ID")
patient_name = st.text_input("Patient Name")
appointment_date = st.date_input("Date", datetime.date.today())

if doctor_id and patient_name:
    st.success("Information entered successfully. Proceed to image upload.")

    # Image upload and model selection
    st.subheader("Step 1: Upload Chest CT Image")
    uploaded_file = st.file_uploader("Upload a Chest CT Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img_array = image_processor.preprocess_image(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Chest CT Image", use_container_width=True)

        st.subheader("Step 2: Select Model(s) for Prediction")
        model_choice = st.radio("Choose a model:", ["U-Net desde Cero", "U-Net Transfer Learning", "Both"])

        if st.button("Generate Prediction"):
            predictions = {}
            metrics_data = {}

            # Run selected model(s)
            if model_choice in ["U-Net desde Cero", "Both"]:
                pred = unet_scratch_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net desde Cero", use_container_width=True)
                predictions["U-Net desde Cero"] = processed_mask
                metrics_data["U-Net desde Cero"] = unet_scratch_history

            if model_choice in ["U-Net Transfer Learning", "Both"]:
                pred = unet_transfer_model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net Transfer Learning", use_container_width=True)
                predictions["U-Net Transfer Learning"] = processed_mask
                metrics_data["U-Net Transfer Learning"] = unet_transfer_history

            # Display metrics
            st.subheader("Model Metrics")
            for model_name, metrics in metrics_data.items():
                st.write(f"**{model_name} Metrics**")
                st.write(f"IoU: {metrics['iou_metric'][-1]:.4f}")
                st.write(f"Dice Coefficient: {metrics['dice_coef'][-1]:.4f}")
                st.write(f"Validation Loss: {metrics['val_loss'][-1]:.4f}")

                # Plot metrics
                fig, ax = plt.subplots()
                ax.plot(metrics['loss'], label="Training Loss")
                ax.plot(metrics['val_loss'], label="Validation Loss")
                ax.plot(metrics['dice_coef'], label="Dice Coefficient")
                ax.plot(metrics['iou_metric'], label="IoU Metric")
                ax.legend()
                st.pyplot(fig)

            # Export option
            if st.button("Export Prediction and Metrics"):
                # Prepare report content
                report = f"""
                Doctor ID: {doctor_id}
                Patient Name: {patient_name}
                Appointment Date: {appointment_date}

                Predictions and Metrics:
                """
                for model_name, metrics in metrics_data.items():
                    report += f"\nModel: {model_name}\n"
                    report += f"IoU: {metrics['iou_metric'][-1]:.4f}\n"
                    report += f"Dice Coefficient: {metrics['dice_coef'][-1]:.4f}\n"
                    report += f"Validation Loss: {metrics['val_loss'][-1]:.4f}\n"

                # Display download button for the report as plain text
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"{patient_name}_prediction_report.txt",
                    mime="text/plain"
                )
else:
    st.warning("Please enter doctor and patient information to proceed.")

