import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from util.utilFunc import isin_center


class Labeller(TransformerMixin, BaseEstimator):
    def fit(self, X):
        return self

    def transform(self, X):
        res = pd.DataFrame(X.groupby("hash").apply(
            lambda series: isin_center(
                series.x_exit.iloc[-1], series.y_exit.iloc[-1]),
        ), columns=["target"])
        return res
