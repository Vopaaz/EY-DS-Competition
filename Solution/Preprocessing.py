from sklearn.preprocessing import StandardScaler
from Coordination import BasePreprocessingExecutor


class StandardPreprocessor(BasePreprocessingExecutor):
    '''
        Wrap the sklearn.preprocessing.StandardPreprocessor.
        NO hyper-parameter is provided. Only use the default parameter.
    '''

    def fit(self, X):
        _, feature, _ = self.split_hash_feature_target(X)
        self.model = StandardScaler()
        self.model.fit(feature)
        return self

    def transform(self, X):
        hash_, feature, target = self.split_hash_feature_target(X)
        res = self.model.transform(feature)
        return self.combine_hash_feature_target(hash_, res, target)
