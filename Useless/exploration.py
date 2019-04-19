import sys
sys.path.append(".")
sys.path.append(r".\Solution")
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from Solution.Coordination import (BaseTrainExecutor,
                                   NanCoordiantor, Submitter)
from Solution.Preprocessing import StandardPreprocessor
from Solution.util.DFPreparation import DFProvider


class ExploreTrainer(BaseTrainExecutor):
    def fit(self, train):
        model = RandomForestClassifier()
        _, feature, target = self.split_hash_feature_target(train)
        return model.fit(feature, target)


if __name__ == "__main__":

    train = DFProvider("train", path_filled=True).get_df()
    test = DFProvider("test", path_filled=True).get_df()

    nc = NanCoordiantor(train, test, "drop")
    nc.preprocess(StandardPreprocessor)
    nc.fit(ExploreTrainer)
    res = nc.predict()

    res = Submitter(res).save("Only for exploration, drop routine, random forest classifier (no hyper-parameter adjusting).")
