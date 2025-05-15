import pandas as pd
import numpy as np
from src.preprocess import (
    replace_ext,
    describe,
    reduce_mem_usage,
    preprocess,
)


def test_replace_ext():
    assert replace_ext("data/train.csv", "parquet") == "data/train.parquet"
    assert replace_ext("data/test", ".json") == "data/test.json"
    assert replace_ext("data/test.csv", "txt") == "data/test.txt"
    assert replace_ext("data.test.csv", ".log") == "data.test.log"


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
    assert desc.loc[desc["feature"] == "C", "null_count"].values[0] == 1
    assert desc.loc[desc["feature"] == "B", "top_value"].values[0] == "a"


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
    assert str(df["ints"].dtype).startswith("int")
    assert str(df["floats"].dtype).startswith("float")


def test_preprocess():
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

    test = train.copy(deep=True)
    preprocess(train, test)

    for col in ["Cabin", "Name", "Ticket", "PassengerId", "Embarked", "Sex"]:
        assert col not in train.columns
        assert col not in test.columns

    assert any(col.startswith("Embarked_") for col in train.columns)
    assert any(col.startswith("Sex_") for col in train.columns)

    assert train.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0

    # Vérifie la consistance des colonnes entre train et test après encodage
    assert set(train.columns) == set(test.columns)

    # Vérifie que le type des colonnes encodées est bien int
    for col in train.columns:
        assert pd.api.types.is_numeric_dtype(train[col])


def test_reduce_mem_usage_type_limits():
    df = pd.DataFrame(
        {
            "int8_col": [np.iinfo(np.int8).min, 0, np.iinfo(np.int8).max],
            "int16_col": [np.iinfo(np.int16).min, 0, np.iinfo(np.int16).max],
            "int32_col": [np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max],
            "int64_col": [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max],
            "float16_col": [np.finfo(np.float16).min, 0.0, np.finfo(np.float16).max],
            "float32_col": [np.finfo(np.float32).min, 0.0, np.finfo(np.float32).max],
            "float64_col": [np.finfo(np.float64).min, 0.0, np.finfo(np.float64).max],
        }
    )
    reduce_mem_usage(df, verbose=False)
    assert df["int8_col"].dtype == np.int8
    assert df["int16_col"].dtype == np.int16
    assert df["int32_col"].dtype == np.int32
    assert df["int64_col"].dtype == np.int64
    assert df["float16_col"].dtype == np.float16
    assert df["float32_col"].dtype == np.float32
    assert df["float64_col"].dtype == np.float64
