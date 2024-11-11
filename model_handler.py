import tensorflow as tf
from metrics import iou_metric, dice_coef, combined_loss  # Import your custom metrics and loss functions

class ModelHandler:
    def __init__(self, model_paths):
        self.models = {}
        for name, path in model_paths.items():
            try:
                # Load model with custom objects and safe_mode=False
                self.models[name] = tf.keras.models.load_model(
                    path,
                    custom_objects={
                        "iou_metric": iou_metric,
                        "dice_coef": dice_coef,
                        "combined_loss": combined_loss
                    },
                    safe_mode=False  # Allow deserialization of models with custom objects or lambda functions
                )
            except Exception as e:
                print(f"Error loading model '{name}' from path '{path}': {e}")
                self.models[name] = None  # Set model to None if it fails to load

    def get_model(self, model_name):
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' could not be loaded. Check for errors in the model path or configuration.")
