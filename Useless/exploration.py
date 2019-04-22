import sys
sys.path.append(".")
sys.path.append(r".\Solution")
from Solution.Training import RandomForestExecutor, GradientBoostingExecutor, SupportVectorExecutor
from Solution.util.DFPreparation import DFProvider
from Solution.Preprocessing import StandardPreprocessor, StandardOutlierPreprocessor
from Solution.Coordination import (BaseTrainExecutor,
                                   NanCoordiantor, Submitter)
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd


class ExploreTrainer(BaseTrainExecutor):
    def fit(self, train):
        model = RandomForestClassifier()
        _, feature, target = self.split_hash_feature_target(train)
        return model.fit(feature, target)


if __name__ == "__main__":

    train = DFProvider("train", path_filled=True).get_df()
    test = DFProvider("test", path_filled=True).get_df()

    # nc = NanCoordiantor(train, test, "fill_0")
    # nc.preprocess(StandardOutlierPreprocessor)
    # nc.fit(RandomForestExecutor)
    # res = nc.predict()
    # Submitter(res).save(
    #     "Only for exploration, RandomForest")

    # nc = NanCoordiantor(train, test, "fill_0")
    # nc.preprocess(StandardOutlierPreprocessor)
    # nc.fit(GradientBoostingExecutor)
    # res = nc.predict()
    # Submitter(res).save(
    #     "Only for exploration, GradientBoosting")

    nc = NanCoordiantor(train.iloc[0:500], test, "fill_0")
    nc.preprocess(StandardOutlierPreprocessor)
    nc.fit(SupportVectorExecutor)
    res = nc.predict()
    Submitter(res).save("1st round SVC params search.")
