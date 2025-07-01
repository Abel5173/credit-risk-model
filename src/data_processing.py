import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import List
# New imports for xverse and woe
from xverse.transformer import WOETransformer as XverseWOETransformer
from xverse.feature_selection import InformationValueThreshold
from woe import WoE

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
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X

class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col + '_le'] = self.encoders[col].transform(X[col].astype(str))
        return X

# --- WOE/IV Transformers ---
class WOEIVTransformer(BaseEstimator, TransformerMixin):
    """
    Uses xverse's WOETransformer for WOE encoding and InformationValueThreshold for IV-based feature selection.
    """
    def __init__(self, target_col='FraudResult', min_iv=0.02):
        self.target_col = target_col
        self.min_iv = min_iv
        self.woe = XverseWOETransformer()
        self.iv_selector = InformationValueThreshold(threshold=self.min_iv)
        self.selected_features_ = None

    def fit(self, X, y=None):
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        X_ = X.drop(columns=[self.target_col], errors='ignore')
        self.woe.fit(X_, y)
        X_woe = self.woe.transform(X_)
        self.iv_selector.fit(X_woe, y)
        self.selected_features_ = self.iv_selector.get_support(indices=True)
        return self

    def transform(self, X):
        X_ = X.drop(columns=[self.target_col], errors='ignore')
        X_woe = self.woe.transform(X_)
        # Select only features above IV threshold
        if self.selected_features_ is not None:
            X_woe = X_woe.iloc[:, self.selected_features_]
        # Add back target if present
        if self.target_col in X.columns:
            X_woe[self.target_col] = X[self.target_col].values
        return X_woe

# --- Main Feature Engineering Pipeline ---
def build_feature_engineering_pipeline(df: pd.DataFrame, scaler: str = 'standard', use_woe_iv: bool = True, min_iv: float = 0.02):
    """
    Builds a robust feature engineering pipeline.
    scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler (normalization)
    use_woe_iv: If True, applies WOE encoding and IV-based feature selection (xverse)
    min_iv: Minimum IV threshold for feature selection
    """
    # Identify columns
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in ['TransactionStartTime', 'CustomerId', 'TransactionId']:
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
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    # Label Encoding for selected columns (example: ProductCategory)
    label_encode_cols = ['ProductCategory'] if 'ProductCategory' in df.columns else []
    label_encoder = LabelEncodingTransformer(label_encode_cols) if label_encode_cols else 'passthrough'
    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('label', label_encoder, label_encode_cols),
    ], remainder='passthrough')
    # Build pipeline steps
    steps = [
        ('aggregate', AggregateFeatures()),
        ('datetime', DateTimeFeatures()),
        ('preprocessor', preprocessor),
    ]
    if use_woe_iv:
        steps.append(('woe_iv', WOEIVTransformer(target_col='FraudResult', min_iv=min_iv)))
    pipeline = Pipeline(steps)
    return pipeline

# Example usage (not for production, just for testing):
# df = pd.read_excel('path_to_data.xlsx')
# pipeline = build_feature_engineering_pipeline(df, scaler='minmax', use_woe_iv=True)
# X_processed = pipeline.fit_transform(df, df['FraudResult'])

