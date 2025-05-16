import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.preprocess import replace_ext, describe, reduce_mem_usage


@pytest.mark.unit
def test_pipeline_reproducibility():
    X = np.array([[1], [2], [np.nan], [4]])
    pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )
    X1 = pipe.fit_transform(X)
    X2 = pipe.transform(X)
    assert np.allclose(X1, X2, equal_nan=True)


# -------------------------------
# Test unitaire : simple remplacement de chaîne
@pytest.mark.unit
def test_replace_ext():
    assert replace_ext("data/train.csv", "parquet") == "data/train.parquet"
    assert replace_ext("data/test", ".json") == "data/test.json"


# -------------------------------
# Test unitaire : vérifie la structure d'un DataFrame de description
@pytest.mark.unit
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
# Test unitaire : optimisation mémoire sur un DataFrame
@pytest.mark.unit
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
