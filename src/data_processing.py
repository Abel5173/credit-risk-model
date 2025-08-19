import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   MinMaxScaler)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
try:
    from xverse.transformer import WOE as XverseWOETransformer
except ImportError:
    from xverse.transformer.woe import WOE as XverseWOETransformer


# --- Custom Transformers ---


class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.customer_id_col)[self.amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('count', 'count'),
            ('std_amount', 'std')
        ]).reset_index()
        X = X.merge(agg, on=self.customer_id_col, how='left')
        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(
            X[self.datetime_col], errors='coerce'
        )
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X


# --- WOE Transformer (no IV selection) ---


class WOEIVTransformer(BaseEstimator, TransformerMixin):
    """
    Uses xverse's WOE transformer for WOE encoding.
    IV-based feature selection is not used.
    """
    def __init__(self, target_col='FraudResult', min_iv=0.02):
        self.target_col = target_col
        self.min_iv = min_iv
        self.woe = XverseWOETransformer()

    def fit(self, X, y=None):
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        X_ = X.drop(columns=[self.target_col], errors='ignore')
        self.woe.fit(X_, y)
        return self

    def transform(self, X):
        X_ = X.drop(columns=[self.target_col], errors='ignore')
        X_woe = self.woe.transform(X_)
        if self.target_col in X.columns:
            X_woe[self.target_col] = X[self.target_col].values
        return X_woe


# --- Main Feature Engineering Pipeline ---


def build_feature_engineering_pipeline(
    df: pd.DataFrame, scaler: str = 'standard', use_woe_iv: bool = True,
    min_iv: float = 0.02
):
    """
    Builds a robust feature engineering pipeline.
    scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
    use_woe_iv: If True, applies WOE encoding (xverse)
    min_iv: (ignored, for compatibility)
    """
    # Identify columns
    numeric_features = df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    for col in [
        'TransactionStartTime',
        'CustomerId',
        'TransactionId',
        'BatchId',
        'AccountId',
        'SubscriptionId',
        'Unnamed: 16',
        'Unnamed: 17',
    ]:
        if col in categorical_features:
            categorical_features.remove(col)
    # Imputation and scaling
    scaler_obj = StandardScaler() if scaler == 'standard' else MinMaxScaler()
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler_obj),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',
                                 sparse_output=False)),
    ])
    # No label encoding for columns already one-hot encoded
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop')  # Drop all other columns
    # Build pipeline steps
    steps = [
        ('aggregate', AggregateFeatures()),
        ('datetime', DateTimeFeatures()),
        ('preprocessor', preprocessor),
    ]
    if use_woe_iv:
        steps.append(
            ('woe_iv', WOEIVTransformer(target_col='FraudResult',
                                        min_iv=min_iv))
        )
    pipeline = Pipeline(steps)
    return pipeline

# Example usage (not for production, just for testing):
# df = pd.read_excel('path_to_data.xlsx')
# pipeline = build_feature_engineering_pipeline(df, scaler='minmax',
#                                               use_woe_iv=True)
# X_processed = pipeline.fit_transform(df, df['FraudResult'])
