import streamlit as st
from model_handler import ModelHandler
from image_processor import ImageProcessor
from history_handler import HistoryHandler
import datetime
import matplotlib.pyplot as plt

# Paths
model_paths = {
    'U-Net desde Cero': "unet_scratch.keras",
    'U-Net Transfer Learning': "unet_transfer.keras"
}
history_paths = {
    'U-Net desde Cero': "unet_scratch_history.npz",
    'U-Net Transfer Learning': "unet_transfer_history.npz"
}

# Initialize handlers
model_handler = ModelHandler(model_paths)
image_processor = ImageProcessor()
history_handler = HistoryHandler(history_paths)

# Set up the interface with a medical theme
st.set_page_config(page_title="Medical Chest CT Segmentation", page_icon="🩺")

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
                model = model_handler.get_model("U-Net desde Cero")
                pred = model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net desde Cero", use_container_width=True)
                predictions["U-Net desde Cero"] = processed_mask
                metrics_data["U-Net desde Cero"] = history_handler.get_metrics("U-Net desde Cero")

            if model_choice in ["U-Net Transfer Learning", "Both"]:
                model = model_handler.get_model("U-Net Transfer Learning")
                pred = model.predict(img_array)
                processed_mask = image_processor.postprocess_mask(pred)
                st.image(processed_mask, caption="Prediction - U-Net Transfer Learning", use_container_width=True)
                predictions["U-Net Transfer Learning"] = processed_mask
                metrics_data["U-Net Transfer Learning"] = history_handler.get_metrics("U-Net Transfer Learning")

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

                # Print to verify report content
                print(report)

                # Display download button for the report as plain text
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"{patient_name}_prediction_report.txt",
                    mime="text/plain"
                )
else:
    st.warning("Please enter doctor and patient information to proceed.")
