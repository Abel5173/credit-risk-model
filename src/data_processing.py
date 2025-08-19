import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from typing import List, Optional
from src.utils import timer_decorator
try:
    from xverse.transformer import WOE as XverseWOETransformer
except ImportError:
    from xverse.transformer.woe import WOE as XverseWOETransformer

# --- Custom Transformers ---


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors='ignore')


class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col: str = 'CustomerId', amount_col: str = 'Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    @timer_decorator
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.customer_id_col not in X.columns or self.amount_col not in X.columns:
            return X
        agg = (
            X.groupby(self.customer_id_col)[self.amount_col]
            .agg([
                ('total_amount', 'sum'),
                ('avg_amount', 'mean'),
                ('count', 'count'),
                ('std_amount', 'std'),
            ])
            .reset_index()
        )
        X = X.merge(agg, on=self.customer_id_col, how='left')
        # std can be NaN for single-observation groups; fill with 0 which is sensible for std
        if 'std_amount' in X.columns:
            X['std_amount'] = X['std_amount'].fillna(0.0)
        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col: str = 'TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    @timer_decorator
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.datetime_col not in X.columns:
            return X
        X[self.datetime_col] = pd.to_datetime(
            X[self.datetime_col], errors='coerce')
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X


class WOETransformer(BaseEstimator, TransformerMixin):
    """WOE encoder using xverse with monotonic binning and missing support.

    Applies WOE to selected columns and returns a DataFrame with WOE features.
    """

    def __init__(self, target_col: str = 'is_high_risk', woe_cols: Optional[List[str]] = None,
                 monotonic_binning: bool = True, woe_missing: bool = True):
        self.target_col = target_col
        self.woe_cols = woe_cols
        self.monotonic_binning = monotonic_binning
        self.woe_missing = woe_missing
        self._woe = XverseWOETransformer(
            monotonic_binning=self.monotonic_binning, woe_missing=self.woe_missing)
        self._fitted = False
        self._cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            if self.target_col in X.columns:
                y = X[self.target_col]
            else:
                raise ValueError(
                    "WOETransformer requires target y or target_col present in X")
        X_fit = X.copy()
        cols = self._resolve_cols(X_fit)
        self._cols_ = cols
        if len(cols) == 0:
            # Nothing to encode; act as a no-op
            self._fitted = True
            return self
        # xverse expects only the columns to encode
        self._woe.fit(X_fit[cols], y)
        self._fitted = True
        return self

    @timer_decorator
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "WOETransformer must be fitted before calling transform")
        if len(self._cols_) == 0:
            return X
        X_tr = X.copy()
        cols = self._cols_
        X_woe_only = self._woe.transform(X_tr[cols])
        remaining = X_tr.drop(
            columns=[c for c in cols if c in X_tr.columns], errors='ignore')
        X_out = pd.concat([remaining, X_woe_only], axis=1)
        return X_out

    def _resolve_cols(self, X: pd.DataFrame) -> List[str]:
        if self.woe_cols is not None:
            return [c for c in self.woe_cols if c in X.columns]
        # Default to frequent categorical-like columns likely to be monotonic binned
        candidates = [
            'ProductCategory', 'ChannelId', 'ProviderId', 'ProductId', 'CurrencyCode'
        ]
        return [c for c in candidates if c in X.columns]


# --- Main Feature Engineering Pipeline ---
def build_feature_engineering_pipeline(
    df: pd.DataFrame, scaler: str = 'standard', use_woe_iv: bool = True
):
    """
    Builds a robust feature engineering pipeline that:
      - adds aggregate and datetime features,
      - applies WOE encoding to key categorical columns (using target is_high_risk),
      - imputes and scales numeric features, one-hot encodes remaining categoricals,
      - preserves column names and passes through unspecified columns.
    """
    scaler_obj = StandardScaler() if scaler == 'standard' else MinMaxScaler()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler_obj),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # Dynamic selectors so engineered columns are included appropriately
    num_sel = make_column_selector(dtype_include=['number'])
    cat_sel = make_column_selector(dtype_include=['object', 'category'])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_sel),
        ('cat', categorical_transformer, cat_sel),
    ], remainder='passthrough')
    # Ensure DataFrame output from ColumnTransformer (scikit-learn >= 1.3)
    try:
        preprocessor.set_output(transform='pandas')
    except Exception:
        pass

    steps = [
        ('drop_target', DropColumns(columns=['is_high_risk'])),
        ('aggregate', AggregateFeatures()),
        ('datetime', DateTimeFeatures()),
    ]

    if use_woe_iv:
        # Choose WOE columns based on current df
        cat_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        woe_candidates = ['ProductCategory', 'ChannelId',
                          'ProviderId', 'ProductId', 'CurrencyCode']
        woe_cols = [c for c in woe_candidates if c in cat_cols]
        steps.append(('woe', WOETransformer(target_col='is_high_risk', woe_cols=woe_cols,
                                            monotonic_binning=True, woe_missing=True)))

    # Drop identifier-like columns and raw timestamp after engineering/WOE
    id_like_cols = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId',
        'CustomerId', 'TransactionStartTime', 'Unnamed: 16', 'Unnamed: 17', 'FraudResult'
    ]
    steps.append(('drop_non_features', DropColumns(columns=id_like_cols)))

    steps.append(('preprocessor', preprocessor))

    pipeline = Pipeline(steps)
    return pipeline

# Example usage (not for production, just for testing):
# df = pd.read_csv('data/processed/processed_data.csv')
# pipeline = build_feature_engineering_pipeline(df, scaler='minmax', use_woe_iv=True)
# X_processed = pipeline.fit_transform(df, df['is_high_risk'])
