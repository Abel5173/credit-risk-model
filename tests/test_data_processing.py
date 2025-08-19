import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

from src.data_processing import AggregateFeatures, DateTimeFeatures


def test_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 300]
    })
    agg = AggregateFeatures()
    result = agg.fit_transform(df)
    assert 'total_amount' in result.columns
    assert result.loc[result['CustomerId'] == 1, 'total_amount'].iloc[0] == 300
    assert result.loc[result['CustomerId'] == 2, 'total_amount'].iloc[0] == 300


def test_datetime_features():
    df = pd.DataFrame({'TransactionStartTime': ['2023-01-01 10:00:00']})
    dt = DateTimeFeatures()
    result = dt.fit_transform(df)
    assert 'transaction_hour' in result.columns
    assert result['transaction_hour'].iloc[0] == 10
