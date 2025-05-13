import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from src import predict  # Assure-toi que le dossier src contient bien un __init__.py
from tests.dummy_model import DummyModel

# ✅ Test fonctionnel : simule le pipeline complet de prédiction, avec mocks
@pytest.mark.functional
@patch("src.predict.load_pickle")
@patch("src.predict.mlflow.search_runs")
@patch("src.predict.pd.read_pickle")
@patch("src.predict.pd.DataFrame.to_csv")
def test_main(mock_to_csv, mock_read_pickle, mock_search_runs, mock_load_pickle):
    # Mock search_runs : simule une réponse MLflow
    mock_search_runs.return_value = pd.DataFrame({
        'params.model_path': ['file:///dummy_model_path'],
        'metrics.accuracy': [0.99],
        'attribute.start_time': ['2023-01-01']
    })

    # Mock load_pickle : retourne des modèles simulés
    mock_load_pickle.return_value = [DummyModel()]

    # Mock read_pickle : retourne un DataFrame d'exemple
    mock_read_pickle.return_value = pd.DataFrame(np.zeros((2, 3)))

    # Appel de la fonction à tester
    predict.main()

    # Vérifie que le fichier CSV a été généré
    mock_to_csv.assert_called_once()