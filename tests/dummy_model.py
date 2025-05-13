import numpy as np

# DummyModel pour simuler un modèle entraîné
class DummyModel:
    def __init__(self):
        self.n_classes_ = 2
        self.best_iteration_ = 0

    def predict_proba(self, X, num_iteration=None):
        return np.array([[0.1, 0.9]] * len(X))