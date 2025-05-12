import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import os
import mlflow
from src.train import divide_by_sum, get_scores, train_model, main
import matplotlib
matplotlib.use('Agg')

# Mock MLflow to avoid making actual calls
@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    # Mocking mlflow methods
    mlflow.create_experiment = MagicMock(return_value=None)
    mlflow.set_experiment = MagicMock()
    # Mock mlflow.start_run to return an object with run_id
    mock_run = MagicMock()
    mock_run.info.run_id = 'mock_run_id'  # Ensure run_id is available
    mlflow.start_run = MagicMock(return_value=mock_run)
    mlflow.log_artifact = MagicMock()
    mlflow.log_metrics = MagicMock()
    mlflow.log_params = MagicMock()
    
    # Mock the UI system call to avoid opening the MLflow UI
    monkeypatch.setattr('os.system', lambda x: None)

# Test pour la fonction divide_by_sum
def test_divide_by_sum():
    series = pd.Series([1, 2, 3, 4, 5])
    result = divide_by_sum(series)
    expected = series / series.sum()
    pd.testing.assert_series_equal(result, expected)

# Test pour la fonction get_scores
def test_get_scores():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    
    scores = get_scores(y_true, y_pred)
    assert 'accuracy' in scores
    assert 'precision' in scores
    assert 'recall' in scores
    assert 'f1' in scores

# Test pour la fonction train_model
def test_train_model():
    # Générer des données factices
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, size=100))

    params = {
        'model': {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'random_state': 0,
            'n_jobs': -1
        },
        'fit': {
            'early_stopping_rounds': 100,
            'verbose': 10
        },
        'fold': {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 0
        }
    }

    # On teste la fonction sans MLflow
    exp_id, run_uuid = train_model(X, y, params, 'titanic')
    
    assert exp_id is not None
    assert run_uuid is not None

# Test pour la fonction main (simuler sans exécuter le script complet)
"""def test_main(monkeypatch):
    monkeypatch.setattr('os.listdir', lambda x: ['data'])
    
    # Mock la lecture de fichier pickle
    train_mock = MagicMock()
    train_mock.drop.return_value = pd.DataFrame(np.random.rand(100, 5))
    train_mock.__getitem__.return_value = pd.Series(np.random.randint(0, 2, 100))
    monkeypatch.setattr(pd, 'read_pickle', lambda path: train_mock)
    
    main()  # Tester la fonction principale"""
