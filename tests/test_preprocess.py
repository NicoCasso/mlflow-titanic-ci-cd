import pandas as pd
import numpy as np
from src.preprocess import replace_ext, describe, reduce_mem_usage, preprocess


# -------------------------------
# Test de replace_ext
# -------------------------------
def test_replace_ext():
    assert replace_ext("data/train.csv", "parquet") == "data/train.parquet"
    assert replace_ext("data/test", ".json") == "data/test.json"


# -------------------------------
# Test de describe
# -------------------------------
def test_describe():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "a", "b", "a"],
            "C": [np.nan, 1, 2, 3, 4],
        }
    )
    desc = describe(df)
    assert "feature" in desc.columns
    assert "null_count" in desc.columns
    assert desc.shape[0] == df.shape[1]


# -------------------------------
# Test de reduce_mem_usage
# -------------------------------
def test_reduce_mem_usage():
    df = pd.DataFrame(
        {
            "ints": np.random.randint(0, 100, size=1000),
            "floats": np.random.rand(1000) * 1000,
        }
    )
    before = df.memory_usage().sum()
    reduce_mem_usage(df, verbose=False)
    after = df.memory_usage().sum()
    assert after <= before


# -------------------------------
# Test de preprocess
# -------------------------------
def test_preprocess():
    # Mini dataset simulé façon Titanic
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Name": ["Name1", "Name2"],
            "Cabin": [np.nan, np.nan],
            "Ticket": ["123", "456"],
            "Embarked": ["C", np.nan],
            "Fare": [7.25, np.nan],
            "Age": [22, np.nan],
            "Sex": ["male", "female"],
            "Pclass": [3, 1],
        }
    )

    test = train.copy()
    preprocess(train, test)

    # Vérifie que les colonnes supprimées ne sont plus là
    for col in ["Cabin", "Name", "Ticket", "PassengerId", "Embarked", "Sex"]:
        assert col not in train.columns
        assert col not in test.columns

    # Vérifie que de nouvelles colonnes ont été créées par OneHotEncoder
    assert any(col.startswith("Embarked_") for col in train.columns)
    assert any(col.startswith("Sex_") for col in train.columns)

    # Vérifie qu'il n'y a plus de valeurs manquantes
    assert train.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0
