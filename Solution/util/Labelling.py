'''
    Label the train set.
'''

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from Solution.util.BaseUtil import isin_center


class Labeller(TransformerMixin, BaseEstimator):
    '''
        Label the train set.
    '''

    def fit(self, X):
        return self

    def transform(self, X):
        '''
            Parameters: a DataFrame containing column "hash", "x_exit", "y_exit".

            Returns: a DataFrame of "hash" numbers of rows, one column "target".
                    The index is the hash value of the device.
        '''
        res = pd.DataFrame(X.groupby("hash").apply(
            lambda series: isin_center(
                series.x_exit.iloc[-1], series.y_exit.iloc[-1]),
        ), columns=["target"])
        return res
