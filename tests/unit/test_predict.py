import numpy as np
import pytest
from src import predict  # Assure-toi que le dossier src contient bien un __init__.py
from tests.dummy_model import DummyModel


# ✅ Test unitaire : vérifie une méthode d'une classe isolée
@pytest.mark.unit
def test_ensemble_model_predict_proba():
    models = [DummyModel(), DummyModel()]
    ensemble = predict.EnsembleModel(models)

    X = np.zeros((3, 4))
    result = ensemble.predict_proba(X)

    expected = np.array([[0.1, 0.9]] * 3)
    np.testing.assert_array_almost_equal(result, expected)
