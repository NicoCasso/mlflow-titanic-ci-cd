import pytest
import pandas as pd
import numpy as np
import os
import mlflow
from src.train import divide_by_sum, get_scores, train_model, main


# -------------------------------
# Test unitaire : opération mathématique simple
@pytest.mark.unit
def test_divide_by_sum():
    series = pd.Series([1, 2, 3, 4, 5])
    result = divide_by_sum(series)
    expected = series / series.sum()
    pd.testing.assert_series_equal(result, expected)

# -------------------------------
# Test unitaire : calcul de scores à partir de y_true / y_pred
@pytest.mark.unit
def test_get_scores():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    
    scores = get_scores(y_true, y_pred)
    assert 'accuracy' in scores
    assert 'precision' in scores
    assert 'recall' in scores
    assert 'f1' in scores


