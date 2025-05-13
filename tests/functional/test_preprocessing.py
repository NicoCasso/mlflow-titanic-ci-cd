import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.preprocess import replace_ext, describe, reduce_mem_usage, preprocess

# -------------------------------
# Test fonctionnel : applique tout un pipeline de preprocessing
@pytest.mark.functional
def test_preprocess():
    # Mini dataset simulé façon Titanic
    train = pd.DataFrame({
        'PassengerId': [1, 2],
        'Name': ['Name1', 'Name2'],
        'Cabin': [np.nan, np.nan],
        'Ticket': ['123', '456'],
        'Embarked': ['C', np.nan],
        'Fare': [7.25, np.nan],
        'Age': [22, np.nan],
        'Sex': ['male', 'female'],
        'Pclass': [3, 1]
    })

    test = train.copy()
    preprocess(train, test)

    # Vérifie que les colonnes supprimées ne sont plus là
    for col in ['Cabin', 'Name', 'Ticket', 'PassengerId', 'Embarked', 'Sex']:
        assert col not in train.columns
        assert col not in test.columns

    # Vérifie que de nouvelles colonnes ont été créées par OneHotEncoder
    assert any(col.startswith('Embarked_') for col in train.columns)
    assert any(col.startswith('Sex_') for col in train.columns)

    # Vérifie qu'il n'y a plus de valeurs manquantes
    assert train.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0