import numpy as np

class HistoryLoader:
    def __init__(self, history_paths):
        self.histories = {name: np.load(path) for name, path in history_paths.items()}

    def get_history(self, model_name):
        return self.histories.get(model_name)
