import sys
sys.path.append(".")
sys.path.append(r".\Solution")

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from Solution.Coordination import BaseTrainExecutor, NanCoordiantor, Submitter
from Solution.Preprocessing import (StandardOutlierPreprocessor,
                                    StandardPreprocessor)
from Solution.Training import (CombinedExecutor, GradientBoostingExecutor,
                               RandomForestExecutor, SupportVectorExecutor,
                               XGBoostExecutor)
from Solution.util.DFPreparation import DFProvider



class ExploreTrainer(BaseTrainExecutor):
    def fit(self, train):
        model = RandomForestClassifier()
        _, feature, target = self.split_hash_feature_target(train)
        return model.fit(feature, target)


if __name__ == "__main__":

    train = DFProvider("train", path_filled=True).get_df()
    test = DFProvider("test", path_filled=True).get_df()

    # nc = NanCoordiantor(train, test, "drop")
    # nc.preprocess(StandardOutlierPreprocessor)
    # nc.fit(RandomForestExecutor)
    # res = nc.predict()
    # Submitter(res).save(
    #     "Drop Strategy, RandomForest")

    # nc = NanCoordiantor(train, test, "drop")
    # nc.preprocess(StandardOutlierPreprocessor)
    # nc.fit(GradientBoostingExecutor)
    # res = nc.predict()
    # Submitter(res).save(
    #     "Drop Strategy, GradientBoosting")

    # nc = NanCoordiantor(train, test, "drop")
    # nc.preprocess(StandardOutlierPreprocessor)
    # nc.fit(SupportVectorExecutor)
    # res = nc.predict()
    # Submitter(res).save("Drop Strategy, SVC")

    # nc = NanCoordiantor(train, test, "drop")
    # nc.preprocess(StandardOutlierPreprocessor)
    # nc.fit(CombinedExecutor)
    # res = nc.predict()
    # Submitter(res).save("Combined Voting without SVC, use only random forest and gradient boosting, drop strategy. Parameters not optimized, using the best parameters of the fill_0 strategy")

    nc = NanCoordiantor(train, test, "drop")
    nc.preprocess(StandardOutlierPreprocessor)
    nc.fit(XGBoostExecutor)
    res = nc.predict()
    Submitter(res).save("XGBOOSTING 1st params")
