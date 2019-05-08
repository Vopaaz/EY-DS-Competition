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

        with open(r"/floyd/input/OriginalFile/data_test/data_test.csv", "r", encoding="utf-8") as f:
            self.test = pd.read_csv(f, index_col=0)

    def __get_raw_train(self):
        r'''
            Read the raw train data table.
            Its path should be r"EY-DS-Competition\OriginalFile\data_train\data_train.csv"
        '''
        with open(r"/floyd/input/OriginalFile/data_train/data_train.csv", "r", encoding="utf-8") as f:
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
    '''
        Return whether the two parameters are both numeric data types.
    '''
    return isinstance(x, (int, float, np.float64, np.int64)) and isinstance(y, (int, float, np.float64, np.int64))


def isin_center(x, y):
    '''
        Return whether a coordinate is in the center of Atlanta.
        The return value will be 1 and 0 instead of True or False,
        so as to be consistent with the competition requirement.

        Parameters can be two single numbers, or two pandas Series.
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
    '''
        Return the l1 distance of a point to the border of the central area.

        Parameters can be two single numbers, or two pandas Series.
        The return value will correspondingly be a number or a Series.
    '''
    if xy_is_number(x, y):
        return _one_point_distance_to_border(x, y)
    elif isinstance(x, pd.Series) and isinstance(y, pd.Series):
        res = pd.DataFrame([x, y]).apply(
            lambda series: _one_point_distance_to_border(series[0], series[1]), axis=0)
        if res.empty:
            return pd.Series([])
        else:
            return res
    else:
        raise TypeError(
            "Parameter type should be both number or both pandas Series. The parameter type now is {}, {}".format(type(x), type(y)))


def _one_point_distance_to_border(x, y):
    '''
        Return the l1 distance of a point to the border of the central area.

        Parameters: Two single numbers
    '''
    if isin_center(x, y):
        d_north = y - MAX_Y
        d_south = MIN_Y - y
        d_east = x - MAX_X
        d_west = MIN_X - x
        return max([d_north, d_south, d_east, d_west])
    else:
        return _one_no_center_point_distance_to_border(x, y)


def _one_no_center_point_distance_to_border(x, y):
    '''
        Return the l1 distance of a point to the border of the central area.
        The point MUST NOT be in the central area.

        Parameters: Two single numbers
    '''
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


def distance_between(x1, y1, x2, y2):
    '''
        Return the l1 distance(s) between the two coodinate (Series).

        Parameters can be four single numbers (representing two coordinates), or four pandas Series.
        The return value will correspondingly be a number or a Series.
    '''
    if xy_is_number(x1, y1) and xy_is_number(x2, y2):
        return _distance_between_points(x1, y1, x2, y2)
    elif isinstance(x1, pd.Series) and isinstance(y1, pd.Series) and isinstance(x2, pd.Series) and isinstance(y2, pd.Series):
        res = pd.DataFrame([x1, y1, x2, y2]).apply(lambda series: _distance_between_points(
            series[0], series[1], series[2], series[3]), axis=0)
        if res.empty:
            return pd.Series([])
        else:
            return res
    else:
        raise TypeError(
            "Parameter type should be all numbers or all pandas Series. The parameter type now is {}, {}, {}, {}".format(type(x1), type(y1), type(x2), type(y2)))


def _distance_between_points(x1, y1, x2, y2):
    '''
        Return the l1 distance(s) between the two coodinate (Series).
    '''
    return abs(x1-x2) + abs(y1-y2)


def time_delta(timestamp1, timestamp2):
    '''
        Return the difference between two pandas Timestamps, counted in seconds.
        The result will always be positive. i.e. The order is not taken into account.

        Parameters can be two pandas Timestamps, or two pandas Series.
        The return value will correspondingly be a number or a Series.
    '''
    if isinstance(timestamp1, pd.Timestamp) and isinstance(timestamp2, pd.Timestamp):
        return _single_pair_time_delta(timestamp1, timestamp2)
    elif isinstance(timestamp1, pd.Series) and isinstance(timestamp2, pd.Series):
        res = pd.DataFrame([timestamp1, timestamp2]).apply(
            lambda series: _single_pair_time_delta(series[0], series[1]), axis=0)
        if res.empty:
            return pd.Series([])
        else:
            return res
    else:
        raise TypeError(
            "Parameter type should be two numbers or two pandas Series. The parameter type now is {}, {}".format(type(timestamp1), type(timestamp2)))


def _single_pair_time_delta(timestamp1, timestamp2):
    '''
        Return the difference between two pandas Timestamps, counted in seconds.
        The result will always be positive. i.e. The order is not taken into account.

        Parameters: Two single pandas Timestamps
    '''
    return abs((timestamp1 - timestamp2).total_seconds())
