'''
    Script that gives the final result.
'''

import sys
sys.path.append(".")

import os
import pandas as pd
from Solution.Machine.Coordination import NanCoordiantor
from Solution.Machine.Preprocessing import StandardOutlierPreprocessor
from Solution.Machine.Training import XGBoostExecutor
from Solution.util.DFPreparation import DFProvider
from Solution.util.Submission import Submitter


def init():
    '''
        Check whether the dataset are put into proper location.
        Create certain directories used in the subsequent approach.
    '''
    dirs = ["Tmp", "Result", "log"]
    src = [r"OriginalFile\data_test\data_test.csv",
           r"OriginalFile\data_train\data_train.csv"]
    try:
        if not (os.path.exists(src[0]) and os.path.exists(src[1])):
            raise Exception("Source Data Not Found.")
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
    except Exception as e:
        raise Exception(
            "Environment Initiallizing Failed. {}".format(e))


def give_result():
    '''
        Run the whole thing and give the result.
    '''
    train = DFProvider("train", path_filled=True).get_df()
    test = DFProvider("test", path_filled=True).get_df()
    nc = NanCoordiantor(train, test, "drop")
    nc.preprocess(StandardOutlierPreprocessor)
    nc.fit(XGBoostExecutor)
    res = nc.predict()
    Submitter(res).save("Final Result.")


if __name__ == "__main__":
    init()
    give_result()
