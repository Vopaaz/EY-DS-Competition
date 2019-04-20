import sys
sys.path.append(".")
sys.path.append(r".\Solution")
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from Solution.Coordination import (BaseTrainExecutor,
                                   NanCoordiantor, Submitter)
from Solution.Preprocessing import StandardPreprocessor, StandardOutlierPreprocessor
from Solution.util.DFPreparation import DFProvider
from Solution.Training import RandomForestExecutor, GradientBoostingExecutor


class ExploreTrainer(BaseTrainExecutor):
    def fit(self, train):
        model = RandomForestClassifier()
        _, feature, target = self.split_hash_feature_target(train)
        return model.fit(feature, target)


if __name__ == "__main__":

    train = DFProvider("train", path_filled=True).get_df().iloc[:100]
    test = DFProvider("test", path_filled=True).get_df()

    nc = NanCoordiantor(train, test, "fill_0")
    nc.preprocess(StandardOutlierPreprocessor)
    nc.fit(RandomForestExecutor)
    res = nc.predict()

    Submitter(res).save(
        "Only for exploration, RandomForest")

    nc = NanCoordiantor(train, test, "fill_0")
    nc.preprocess(StandardOutlierPreprocessor)
    nc.fit(GradientBoostingExecutor)
    res = nc.predict()
    Submitter(res).save(
        "Only for exploration, GradientBoosting")
