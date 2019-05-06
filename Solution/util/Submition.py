import datetime
import os

import pandas as pd

from Solution.util.BaseUtil import Raw_DF_Reader


class Submitter(object):
    '''
        Convert the hash_target DataFrame to the (traj_)id_target DataFrame,
        which is the required format.
        It can also save the result and with some memo infomation.

        Parameters:
            - hash_result: the prediction result by other components.
                Should have two columns: "hash" and "target"

        Attributes:
            - result: the transformed DataFrame with columns "id" and "target"
    '''

    def __init__(self, hash_result):
        self.hash_result = hash_result
        self.__transform_result()

    def __transform_result(self):
        '''
            Transform the hash_target DataFrame to
        '''
        raw_test = Raw_DF_Reader().test
        groups = raw_test.groupby("hash")
        result = pd.DataFrame()
        result["id"] = self.hash_result.apply(
            lambda series: groups.get_group(series.hash).trajectory_id.iloc[-1], axis=1)
        result["target"] = self.hash_result.target
        self.result = result

    def save(self, memo=""):
        '''
            Save the result DataFrame to csv file.
            The target diretory is "Result". The file will be named by monthday-hour-minute-second.

            Parameters:
                - memo: A string that describes this result DataFrame, it will be written in the memo.txt under the Result dir.
        '''
        filename = datetime.datetime.now().strftime(r"%m%d-%H-%M-%S") + ".csv"
        filepath = os.path.join("Result", filename)
        self.result.to_csv(filepath, encoding="utf-8",
                           index=False, line_terminator="\n")

        with open(os.path.join("Result", "memo.txt"), "a+", encoding="utf-8") as f:
            f.write(filename)
            f.write("\t")
            f.write(str(memo))
            f.write("\n")
