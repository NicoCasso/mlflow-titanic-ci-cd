import numpy as np
import pandas as pd
from unittest.mock import patch
from src import (
    predict,
)  # assure-toi que le dossier src est bien un module Python (__init__.py)


# DummyModel pour simuler un modèle entraîné
class DummyModel:
    def __init__(self):
        self.n_classes_ = 2
        self.best_iteration_ = 0

    def predict_proba(self, X, num_iteration=None):
        return np.array([[0.1, 0.9]] * len(X))


# Test de la classe EnsembleModel
def test_ensemble_model_predict_proba():
    models = [DummyModel(), DummyModel()]
    ensemble = predict.EnsembleModel(models)

    X = np.zeros((3, 4))
    result = ensemble.predict_proba(X)

    expected = np.array([[0.1, 0.9]] * 3)
    np.testing.assert_array_almost_equal(result, expected)


# Test de la fonction main
@patch("src.predict.load_pickle")
@patch("src.predict.mlflow.search_runs")
@patch("src.predict.pd.read_pickle")
@patch("src.predict.pd.DataFrame.to_csv")
def test_main(mock_to_csv, mock_read_pickle, mock_search_runs, mock_load_pickle):
    # Mock search_runs : simule une réponse MLflow
    mock_search_runs.return_value = pd.DataFrame(
        {
            "params.model_path": ["file:///dummy_model_path"],
            "metrics.accuracy": [0.99],
            "attribute.start_time": ["2023-01-01"],
        }
    )

    # Mock load_pickle : retourne des modèles simulés
    mock_load_pickle.return_value = [DummyModel()]

    # Mock read_pickle : retourne un DataFrame d'exemple
    mock_read_pickle.return_value = pd.DataFrame(np.zeros((2, 3)))

    # Appel de la fonction à tester
    predict.main()

    # Vérifie que le fichier CSV a été généré
    mock_to_csv.assert_called_once()
