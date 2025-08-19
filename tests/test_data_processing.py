import sys
import os
import pandas as pd
from src.data_processing import AggregateFeatures, DateTimeFeatures, build_feature_engineering_pipeline

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


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


def test_pipeline_with_woe_fit_transform():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2],
        'Amount': [100, 200, 50, 80],
        'TransactionStartTime': ['2024-01-01', '2024-01-02', '2024-01-05', '2024-01-06'],
        'ProductCategory': ['A', 'B', 'A', 'B'],
        'ChannelId': ['Online', 'Store', 'Online', 'Store'],
        'is_high_risk': [0, 0, 1, 1]
    })
    pipe = build_feature_engineering_pipeline(
        df, scaler='standard', use_woe_iv=True)
    X_fe = pipe.fit_transform(df, df['is_high_risk'])
    # Ensure we got a 2D array-like with no NaNs
    assert hasattr(X_fe, 'shape') and X_fe.shape[0] == len(df)
    if isinstance(X_fe, pd.DataFrame):
        assert not X_fe.isna().any().any()
        # Heuristic check that WOE applied
        assert any('WOE' in str(c) for c in X_fe.columns)
