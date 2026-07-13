import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=0.01, upper_percentile=0.99, factor=1.5):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        Q1 = X_df.quantile(self.lower_percentile)
        Q3 = X_df.quantile(self.upper_percentile)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X)
        X_clipped = X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X_clipped
