import pytest
import numpy as np
import pandas as pd
import os
from tempfile import NamedTemporaryFile
import matplotlib
matplotlib.use('Agg')

from src.plot_funcs import (
    label_share, corr_matrix, confusion_matrix,
    metric, feature_importance, scores,
    roc_curve, pr_curve
)

def temp_filepath():
    """Utilitaire pour créer un chemin temporaire pour une image."""
    f = NamedTemporaryFile(delete=False, suffix=".png")
    f.close()
    return f.name

@pytest.mark.unit
def test_label_share():
    share = pd.Series([10, 20, 30], index=['A', 'B', 'C'])
    fp = temp_filepath()
    label_share(share, fp)
    assert os.path.exists(fp)
    os.remove(fp)

@pytest.mark.unit
def test_corr_matrix():
    data = np.random.rand(5, 5)
    corr = pd.DataFrame(data).corr()
    fp = temp_filepath()
    corr_matrix(corr, fp)
    assert os.path.exists(fp)
    os.remove(fp)

@pytest.mark.unit
def test_confusion_matrix():
    cm = np.array([[50, 10],
                   [5, 35]])
    fp = temp_filepath()
    confusion_matrix(cm, fp)
    assert os.path.exists(fp)
    os.remove(fp)

@pytest.mark.unit
def test_metric():
    metrics = [
        {'name': 'Accuracy', 'values': [0.6, 0.7, 0.8, 0.75], 'best_iteration': 3},
        {'name': 'Accuracy', 'values': [0.65, 0.68, 0.73, 0.72], 'best_iteration': 3},
    ]
    fp = temp_filepath()
    metric(metrics, fp)
    assert os.path.exists(fp)
    os.remove(fp)

@pytest.mark.unit
def test_feature_importance():
    features = np.array(['f1', 'f2', 'f3'])
    importances = np.array([0.2, 0.5, 0.3])
    fp = temp_filepath()
    feature_importance(features, importances, "Feature Importance", fp)
    assert os.path.exists(fp)
    os.remove(fp)

@pytest.mark.unit
def test_scores():
    scores_dict = {'acc': 0.92, 'f1': 0.85, 'recall': 0.88, 'precision': 0.89}
    fp = temp_filepath()
    scores(scores_dict, fp)
    assert os.path.exists(fp)
    os.remove(fp)

@pytest.mark.unit
def test_roc_curve():
    fpr = np.array([0.0, 0.1, 0.2, 1.0])
    tpr = np.array([0.0, 0.4, 0.8, 1.0])
    auc = 0.85
    fp = temp_filepath()
    roc_curve(fpr, tpr, auc, fp)
    assert os.path.exists(fp)
    os.remove(fp)

@pytest.mark.unit
def test_pr_curve():
    prec = np.array([1.0, 0.8, 0.6, 0.4])
    rec = np.array([0.0, 0.2, 0.4, 1.0])
    auc = 0.78
    fp = temp_filepath()
    pr_curve(prec, rec, auc, fp)
    assert os.path.exists(fp)
    os.remove(fp)
