import sys
sys.path.append(".")

from Solution.util.BaseUtil import Raw_DF_Reader
from Solution.util.Submition import Submitter
from Solution.Machine.Training import XGBoostExecutor
from Solution.util.DFPreparation import DFProvider
from Solution.Machine.Coordination import NanCoordiantor
from Solution.Machine.Preprocessing import StandardOutlierPreprocessor

if __name__ == "__main__":
    train = DFProvider("train").get_df().iloc[:100]
    test = DFProvider("test").get_df().iloc[:100]

    nc = NanCoordiantor(train, test, "drop")
    nc.preprocess(StandardOutlierPreprocessor)
    nc.fit(XGBoostExecutor)
    res = nc.predict()
    Submitter(res).save("XGBOOSTING 1st params")
