import pandas as pd


class Raw_DF_Reader(object):
    '''
        Provide the raw test/train dataframe.
        Attributes:
            test: Test dataset
            train: Train dataset
        In the table, "time_entry" and "time_exit" column are datetime data type,
        their year-month-date will be 1900-01-01 as they it is not provided in the source.
    '''

    def __init__(self):
        self.__get_raw_test()
        self.__get_raw_train()
        self.__preprocess()

    def __get_raw_test(self):
        r'''
            Read the raw test data table.
            Its path should be r"EY-DS-Competition\OriginalFile\data_test\data_test.csv"
        '''

        with open(r"OriginalFile\data_test\data_test.csv", "r", encoding="utf-8") as f:
            self.test = pd.read_csv(f, index_col=0)

    def __get_raw_train(self):
        r'''
            Read the raw train data table.
            Its path should be r"EY-DS-Competition\OriginalFile\data_train\data_train.csv"
        '''
        with open(r"OriginalFile\data_train\data_train.csv", "r", encoding="utf-8") as f:
            self.train = pd.read_csv(f, index_col=0)

    def __preprocess(self):
        '''
            Convert the "time_entry" and "time_exit" column into datetime data type.
        '''

        self.test.loc[:, ["time_entry", "time_exit"]] = pd.to_datetime(
            self.test[["time_entry", "time_exit"]].stack(), format=r"%H:%M:%S").unstack()

        self.train.loc[:, ["time_entry", "time_exit"]] = pd.to_datetime(
            self.train[["time_entry", "time_exit"]].stack(), format=r"%H:%M:%S").unstack()


def isin_center(x, y):
    '''
        Return whether a coordinate is in the center of Atlanta.
        The return value will be 1 and 0 instead of True or False,
        so as to be consistent with the competition requirement.

        The parameters can be two single numbers, or two pandas Series.
        The return value will correspondingly be a number or a Series consists of 1 and 0.
    '''

    MIN_X = 3750901.5068
    MAX_X = 3770901.5068
    MIN_Y = -19268905.6133
    MAX_Y = -19208905.6133

    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        res = MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y
        return 1 if res else 0
    elif isinstance(x, pd.Series) and isinstance(y, pd.Series):
        res = (MIN_X <= x) & (x <= MAX_X) & (MIN_Y <= y) & (y <= MAX_Y)
        res = res.apply(lambda x: 1 if x else 0)
        res.name = "target"   # To make it in accordance with the submission file
        return res
    else:
        raise TypeError(
            "Parameter type should be both number or both pandas Series. The parameter type now is {}, {}".format(type(x), type(y)))


'''
# Tests
if __name__ == "__main__":
    print(isin_center(0, 0))
    print(isin_center(3750901.5069, -19268905.6132))
    x = pd.Series([0, 3750901.5069, 123])
    y = pd.Series([0, -19268905.6132, -123])
    print(isin_center(x, y))
'''
