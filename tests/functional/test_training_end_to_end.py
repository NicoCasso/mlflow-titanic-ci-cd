import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

@pytest.mark.functional
def test_model_convergence():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    assert model.n_iter_[0] < 1000

