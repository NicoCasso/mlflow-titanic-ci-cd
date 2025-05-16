import pytest
import pandas as pd


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 30, 35, None],
        'gender': ['M', 'F', 'F', 'M'],
        'income': [50000, 60000, 1200000, 55000],
    })

@pytest.mark.functional
def test_required_columns(sample_data):
    required_cols = ['age', 'gender', 'income']
    assert all(col in sample_data.columns for col in required_cols)

@pytest.mark.functional
def test_data_types(sample_data):
    assert pd.api.types.is_numeric_dtype(sample_data['age'])
    assert pd.api.types.is_object_dtype(sample_data['gender'])

@pytest.mark.functional
def test_missing_values(sample_data):
    assert sample_data.isnull().sum().sum() > 0

@pytest.mark.functional
def test_outliers_income(sample_data):
    assert (sample_data['income'] > 1_000_000).sum() == 1
