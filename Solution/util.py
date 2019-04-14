import pandas as pd
import numpy as np

MIN_X = 3750901.5068
MAX_X = 3770901.5068
MIN_Y = -19268905.6133
MAX_Y = -19208905.6133


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


def xy_is_number(x, y):
    return isinstance(x, (int, float, np.float64, np.int64)) and isinstance(y, (int, float, np.float64, np.int64))


def isin_center(x, y):
    '''
        Return whether a coordinate is in the center of Atlanta.
        The return value will be 1 and 0 instead of True or False,
        so as to be consistent with the competition requirement.

        The parameters can be two single numbers, or two pandas Series.
        The return value will correspondingly be a number or a Series consists of 1 and 0.
    '''

    if xy_is_number(x, y):
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


def distance_to_border(x, y):
    if xy_is_number(x, y):
        return _one_point_distance_to_border(x, y)
    elif isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return pd.DataFrame([x, y]).T.apply(
            lambda series: _one_point_distance_to_border(series[0], series[1]), axis=1)
    else:
        raise TypeError(
            "Parameter type should be both number or both pandas Series. The parameter type now is {}, {}".format(type(x), type(y)))


def _one_point_distance_to_border(x, y):
    if isin_center(x, y):
        d_north = y - MAX_Y
        d_south = MIN_Y - y
        d_east = x - MAX_X
        d_west = MIN_X - x
        return max([d_north, d_south, d_east, d_west])
    else:
        return _one_no_center_point_distance_to_border(x, y)


def _one_no_center_point_distance_to_border(x, y):
    if MIN_X <= x <= MAX_X:
        return min([abs(y-MIN_Y), abs(y-MAX_Y)])
    elif MIN_Y <= y <= MAX_Y:
        return min([abs(x-MIN_X), abs(x-MAX_X)])
    else:   # The four corner
        d1 = abs(x-MIN_X)+abs(y-MIN_Y)
        d2 = abs(x-MIN_X)+abs(y-MAX_Y)
        d3 = abs(x-MAX_X)+abs(y-MIN_Y)
        d4 = abs(x-MAX_X)+abs(y-MAX_Y)
        return min([d1, d2, d3, d4])
