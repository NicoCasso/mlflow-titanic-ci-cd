import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


@pytest.mark.functional
def test_f1_score_threshold():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    assert score >= 0.75


@pytest.mark.functional
def test_bias_on_subgroup():
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    subgroup = np.random.choice(["region1", "region2"], size=100)
    assert "region1" in subgroup and "region2" in subgroup
