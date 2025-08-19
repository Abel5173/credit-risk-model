import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from src.data_processing import build_feature_engineering_pipeline
from src.model_utils import calculate_credit_score, predict_optimal_loan


def test_end_to_end_small_sample():
    # Synthetic minimal dataset
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3, 3],
        'Amount': [100, 50, 200, 10, 500, 5],
        'TransactionStartTime': ['2024-01-01', '2024-01-02', '2024-01-05', '2024-01-10', '2024-02-01', '2024-02-02'],
        'ProductCategory': ['A', 'A', 'B', 'B', 'C', 'C'],
        'ChannelId': ['Online', 'Store', 'Online', 'Store', 'Online', 'Store'],
        'is_high_risk': [0, 0, 1, 1, 0, 1]
    })
    y = df['is_high_risk'].astype(int)
    X = df.drop(columns=['is_high_risk'])

    fe = build_feature_engineering_pipeline(df, use_woe_iv=True)
    pipe = SkPipeline([
        ('features', fe),
        ('clf', LogisticRegression(max_iter=500))
    ])
    pipe.fit(X, y)

    probs = pipe.predict_proba(X)[:, 1]
    assert len(probs) == len(X)

    scores = calculate_credit_score(probs)
    assert scores.shape[0] == len(X)
    # Loans for mean probability
    amt, dur = predict_optimal_loan(float(np.mean(probs)))
    assert isinstance(amt, float) and isinstance(dur, int)
