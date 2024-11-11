import streamlit as st
from model_handler import ModelHandler
from history_handler import HistoryHandler
from image_processor import ImageProcessor
import numpy as np

# Paths to models and history files
model_paths = {
    'U-Net desde Cero': "C:\\Users\\Lenovo\\Desktop\\Semestre 7\\DIP\\Modelos_funciona_transfer\\unet_scratch.keras",
    'U-Net Transfer Learning': "C:\\Users\\Lenovo\\Desktop\\Semestre 7\\DIP\\Modelos_funciona_transfer\\unet_transfer.keras"
}
history_paths = {
    'U-Net desde Cero': "C:\\Users\\Lenovo\\Desktop\\Semestre 7\\DIP\\Modelos_funciona_transfer\\unet_scratch_history.npz",
    'U-Net Transfer Learning': "C:\\Users\\Lenovo\\Desktop\\Semestre 7\\DIP\\Modelos_funciona_transfer\\unet_transfer_history.npz"
}

# Initialize handlers
model_handler = ModelHandler(model_paths)
history_handler = HistoryHandler(history_paths)
image_processor = ImageProcessor(target_size=(128, 128))

st.title("Medical Image Segmentation")

uploaded_file = st.file_uploader("Upload a Chest CT image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display original image
    st.image(uploaded_file, caption="Imagen Original", use_column_width=True)

    # Preprocess image
    img_array = image_processor.preprocess_image(uploaded_file)

    # Select model
    model_name = st.selectbox("Selecciona un modelo", list(model_paths.keys()))
    model = model_handler.get_model(model_name)

    if model is not None:
        # Predict segmentation mask
        prediction = model.predict(img_array)
        processed_mask = image_processor.postprocess_mask(prediction)

        # Display the processed mask
        st.image(processed_mask, caption="Predicción de Segmentación", use_column_width=True)

        # Display additional metrics from history
        history_metrics = history_handler.get_history(model_name)

        if history_metrics:
            st.subheader(f"Historial de Métricas para {model_name}")

            # Dynamically check for available metrics
            if 'val_iou_metric' in history_metrics:
                st.write(f"**IoU Final**: {history_metrics['val_iou_metric'][-1]:.4f}")
            else:
                st.write("**IoU Final**: No disponible")

            if 'val_dice_coef' in history_metrics:
                st.write(f"**Dice Coefficient Final**: {history_metrics['val_dice_coef'][-1]:.4f}")
            else:
                st.write("**Dice Coefficient Final**: No disponible")

            if 'val_loss' in history_metrics:
                st.write(f"**Pérdida Final**: {history_metrics['val_loss'][-1]:.4f}")
            else:
                st.write("**Pérdida Final**: No disponible")
