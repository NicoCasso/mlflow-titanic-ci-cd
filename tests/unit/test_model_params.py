import pytest
from sklearn.linear_model import LogisticRegression

@pytest.mark.unit
def test_hyperparameter_application():
    model = LogisticRegression(C=0.5)
    assert model.get_params()['C'] == 0.5