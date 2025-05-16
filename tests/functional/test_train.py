import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import mlflow
from src.train import train_model


# -------------------------------
# Fixture : mock de MLflow pour tous les tests
@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    mlflow.create_experiment = MagicMock(return_value=None)
    mlflow.set_experiment = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "mock_run_id"
    mlflow.start_run = MagicMock(return_value=mock_run)
    mlflow.log_artifact = MagicMock()
    mlflow.log_metrics = MagicMock()
    mlflow.log_params = MagicMock()
    monkeypatch.setattr("os.system", lambda x: None)


# -------------------------------
# Test fonctionnel : exécution complète de l'entraînement d'un modèle
@pytest.mark.functional
def test_train_model():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, size=100))

    params = {
        "model": {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": 0,
            "n_jobs": -1,
        },
        "fit": {"early_stopping_rounds": 100, "verbose": 10},
        "fold": {"n_splits": 5, "shuffle": True, "random_state": 0},
    }

    exp_id, run_uuid = train_model(X, y, params, "titanic")

    assert exp_id is not None
    assert run_uuid is not None


# -------------------------------
# Test fonctionnel (désactivé ici) : pipeline complet de train
# Pour l’activer, décommenter la fonction

# @pytest.mark.functional
# def test_main(monkeypatch):
#     monkeypatch.setattr('os.listdir', lambda x: ['data'])

#     train_mock = MagicMock()
#     train_mock.drop.return_value = pd.DataFrame(np.random.rand(100, 5))
#     train_mock.__getitem__.return_value = pd.Series(np.random.randint(0, 2, 100))
#     monkeypatch.setattr(pd, 'read_pickle', lambda path: train_mock)

#     main()
