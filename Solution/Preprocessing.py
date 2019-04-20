from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from Coordination import BasePreprocessingExecutor
import pandas as pd


class StandardPreprocessor(BasePreprocessingExecutor):
    '''
        Wrap the sklearn.preprocessing.StandardPreprocessor.
        NO hyper-parameter is provided. Only use the default parameter.
    '''

    def fit(self, X):
        _, feature, _ = self.split_hash_feature_target(X)
        self.std_scaler = StandardScaler()
        self.std_scaler.fit(feature)
        return self

    def transform(self, X):
        hash_, feature, target = self.split_hash_feature_target(X)
        res = self.std_scaler.transform(feature)
        return self.combine_hash_feature_target(hash_, res, target)


class StandardOutlierPreprocessor(BasePreprocessingExecutor):
    '''
        Wrap the sklearn.preprocessing.StandardPreprocessor and sklearn.ensemble.IsolationForest
        Parameter (please put it in kwargs when initiallizing)
            - contamination: IsolationForest contamination parameter.
            Refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest.
    '''

    def fit(self, X):
        _, feature, _ = self.split_hash_feature_target(X)
        contamination = self.kwargs["contamination"] if "contamination" in self.kwargs else 0.05
        self.i_forest = IsolationForest(
            contamination=contamination, behaviour="new")
        self.std_scaler = StandardScaler()
        self.i_forest.fit(feature)
        self.std_scaler.fit(feature)
        return self

    def transform(self, X):
        if "":   # Case train set.
            pred_result = self.i_forest.predict(X)
            X = X[pred_result == 1]
        hash_, feature, target = self.split_hash_feature_target(X)
        res = self.std_scaler.transform(feature)
        return self.combine_hash_feature_target(hash_, res, target)
