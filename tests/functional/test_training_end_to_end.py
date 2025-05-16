import pytest
from sklearn.datasets import make_classification
import lightgbm as lgbm

@pytest.mark.functional
def test_model_convergence():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    model = lgbm.LGBMClassifier(n_estimators=100, verbosity=-1)
    model.fit(X, y)
    
    # On vérifie que le modèle a bien appris (score > 0.8 par exemple)
    score = model.score(X, y)
    assert score > 0.8
