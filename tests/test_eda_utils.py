import os
import pandas as pd
import pytest

from eda_utils import load_data, clean_data, engineer_features


def test_load_and_clean(tmp_path):
    # use a tiny dataframe saved to excel to test load/clean
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2021-01-01', '2021-01-02']),
        'Price per Unit': [10, 20],
        'Quantity': [1, 2],
        'Total Amount': [10, 40],
        'Age': [30, 40],
        'Gender': ['Male', 'Female'],
        'Product Category': ['A', 'B'],
        'Transaction ID': [1, 2]
    })
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)

    loaded = load_data(str(path))
    assert loaded.shape == (2, df.shape[1])

    cleaned = clean_data(loaded)
    # cleaning should not change row count
    assert cleaned.shape[0] == 2

    feat = engineer_features(cleaned)
    assert 'Year' in feat.columns
    assert 'Age_Group' in feat.columns
    assert feat['Gender_Enc'].tolist() == [1, 0]


def test_engineer_features_missing_columns():
    # ensure function raises or handles missing expected columns gracefully
    with pytest.raises(KeyError):
        engineer_features(pd.DataFrame())
