# history_handler.py
import numpy as np

class HistoryHandler:
    def __init__(self, history_paths):
        self.history_data = self._load_histories(history_paths)

    def _load_histories(self, history_paths):
        history_data = {}
        for model_name, path in history_paths.items():
            try:
                data = np.load(path)
                # Cargar solo las métricas necesarias
                history_data[model_name] = {
                    'loss': data['loss'],
                    'val_loss': data['val_loss'],
                    'dice_coef': data['dice_coef'],
                    'iou_metric': data['iou_metric']
                }
            except KeyError as e:
                print(f"Error: La métrica esperada {e} no está en el archivo {path}")
                history_data[model_name] = {}
            except Exception as e:
                print(f"No se pudo cargar el historial para el modelo '{model_name}' desde '{path}': {e}")
                history_data[model_name] = {}
        return history_data

    def get_metrics(self, model_name):
        return self.history_data.get(model_name, {})

