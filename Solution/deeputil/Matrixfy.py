import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from Solution.util.BaseUtil import time_delta
from Solution.deeputil.ValueFunc import naive_value


time_entry_ix = 2
time_exit_ix = 3
x_entry_ix = 7
y_entry_ix = 8
x_exit_ix = 9
y_exit_ix = 10


class Path(object):
    '''
        Contains time and location information of a path
        Parameters:
            - i_start: the vertical position of start point in the map
            - j_start: the horizontal position of start point in the map
            - i_end: the vertical position of end point in the map
            - j_end: the horizontal position of end point in the map
            - sPoint_x: x coordination of the start point
            - sPoint_y: y coordination of the start point
            - ePoint_x: x coordination of the end point
            - ePoint_y: y coordination of the end point
            - start_time: the start time
            - end_time: the end time
    '''

    def __init__(self, i_start, j_start, i_end, j_end, sPoint_x, sPoint_y, ePoint_x, ePoint_y, start_time, end_time):
        self.i_start = i_start
        self.j_start = j_start
        self.i_end = i_end
        self.j_end = j_end
        self.sPoint_x = sPoint_x
        self.sPoint_y = sPoint_y
        self.ePoint_x = ePoint_x
        self.ePoint_y = ePoint_y
        self.start_time = start_time
        self.end_time = end_time


def _get_dist(point_x, point_y, line_x1, line_y1, line_x2, line_y2):
    '''
        Parameters:
            - point_x: x coordination of the point
            - point_y: y coordination of the point
            - line_x1: x coordination of the start point of the line
            - line_y1: y coordination of the start point of the line
            - line_x2: x coordination of the end point of the line
            - line_y2: y coordination of the end point of the line
        Return:
             The l2 distance of the point to the line
    '''
    a = line_y2 - line_y1
    b = line_x1 - line_x2
    c = line_x2 * line_y1 - line_x1 * line_y2
    dis = (math.fabs(a*point_x+b*point_y+c))/(math.pow(a*a+b*b, 0.5))
    return dis


def _point_dist(x1, y1, x2, y2):
    '''
        Parameters:
            - x1: the x coordination of point1
            - y1: the y coordination of point1
            - x2: the x coordination of point2
            - y2: the y coordination of point2
        Return:
             The l2 distance from one point to the other
    '''
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def _position_case(sPoint_x, sPoint_y, ePoint_x, ePoint_y):
    '''
        Determine which kind of relative position the 2 point is in
        Parameters:
            - path.sPoint_x: the x coordination of the start point
            - sPoint_y: the y coordination of the start point
            - ePoint_x: the x coordination of the end point
            - path.ePoint_y: the y coordination of the end point
        Returns:
            0 represents right-down
            1 represents left-down
            2 represents left-up
            3 represents right-up
            -1 represents the same point
    '''
    if sPoint_x < ePoint_x and sPoint_y <= ePoint_y:
        return 0
    elif sPoint_x >= ePoint_x and sPoint_y < ePoint_y:
        return 3
    elif sPoint_x > ePoint_x and sPoint_y >= ePoint_y:
        return 2
    elif sPoint_x <= ePoint_x and sPoint_y > ePoint_y:
        return 1
    elif sPoint_x == ePoint_x and sPoint_y == ePoint_y:
        return -1


def _next_place(i, j, case, d1, d2, d3, d4):
    '''
        Select next square according to the distance
        Parameters:
            - i: the vertical position of the point in the map
            - j: the horizontal position of the point in the map
            - case: the situation of position that the point is in
            - d1: distance1 from MatrixfyTransformer.__matrix_path
            - d2: distance2 from MatrixfyTransformer.__matrix_path
            - d3: distance3 from MatrixfyTransformer.__matrix_path
            - d4: distance4 from MatrixfyTransformer.__matrix_path
        Returns:
            The next matrix place (i, j)
    '''
    if case == 1:
        if d1 < d4:
            return i+1, j
        else:
            return i, j-1
    elif case == 2:
        if d3 < d4:
            return i-1, j
        else:
            return i, j-1
    elif case == 3:
        if d3 < d2:
            return i-1, j
        else:
            return i, j+1
    elif case == 0:
        if d1 < d2:
            return i+1, j
        else:
            return i, j+1


class MatrixfyTransformer(TransformerMixin, BaseEstimator):
    '''
        To transform the data to a matrix map
        Parameters:
            - pixel: representing the width and height for one pixel in the map
            - value_func: the value assign function for pixels in the map

        Attributes:
            min_x: the minimum x coordination of train & test
            max_x: the maximum x coordination of train & test
            min_y: the minimum y coordination of train & test
            max_y: the maximum x coordination of train & test
            resolution: the number of pixels in height and width

    '''
    def __init__(self, pixel=1000, value_func=naive_value):
        self.pixel = pixel
        self.value_func = value_func

    def fit(self, train, test):
        self.min_x = min(train.x_entry.min(), train.x_exit.min(),
                         test.x_entry.min(), test.x_exit.min())
        self.max_x = max(train.x_entry.max(), train.x_exit.max(),
                         test.x_entry.max(), test.x_exit.max())

        self.min_y = min(train.y_entry.min(), train.y_exit.min(),
                         test.y_entry.min(), test.y_exit.min())
        self.max_y = max(train.y_entry.max(), train.y_exit.max(),
                         test.y_entry.max(), test.y_exit.max())

        self.resolution = (
            math.floor((self.max_x - self.min_x)/self.pixel) + 1,
            math.floor((self.max_y - self.min_y)/self.pixel) + 1
        )

        return self

    def transform(self, X):
        return pd.DataFrame(X.groupby("hash").apply(self.__matrixfy_one_device), columns=["map_"])

    def __center_x(self, i):
        return (i + 0.5) * self.pixel + self.min_x

    def __center_y(self, j):
        return (j + 0.5) * self.pixel + self.min_y

    def __xy_to_ij(self, point_x, point_y):
        '''
            Determine which square the point is in
            Parameters:
                - point_x: the x coordination of the point
                - point_y: the y coordination of the point
                - pixel: the size of one square
            Returns:
                The position of the point in the matrix. (like (i, j))
        '''
        return int((point_x - self.min_x) / self.pixel), int((point_y - self.min_y) / self.pixel)

    def __assign_value(self, i, j, path):
        '''
            Assign value to the selected square
            Return:
                The value to be assigned to the selected square
        '''
        start_dist = _point_dist(self.__center_x(
            i), self.__center_y(j), path.sPoint_x, path.sPoint_y)
        end_dist = _point_dist(self.__center_x(
            i), self.__center_y(j), path.ePoint_x, path.ePoint_y)

        ratio = start_dist / (start_dist + end_dist)
        base_time = datetime.strptime(
            '1900-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

        delta = path.end_time - path.start_time
        this_time_timestamp = path.start_time + delta*ratio
        value_number = self.value_func(this_time_timestamp)

        return value_number

    def __matrix_path(self, map_, path, case):
        '''
            The main function to construct the matrix
            Return:
                The completed matrix path
                The queue that contains information of row, column and value
        '''
        i, j = path.i_start, path.j_start
        while (not ((i == path.i_end) and (j == path.j_end))):
            i, j = path.i_start, path.j_start
            map_[i, j] = self.__assign_value(i, j, path)
            d1 = _get_dist(self.__center_x(i + 1), self.__center_y(j),
                           path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # down
            d2 = _get_dist(self.__center_x(i), self.__center_y(j + 1),
                           path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # right
            d3 = _get_dist(self.__center_x(i - 1), self.__center_y(j),
                           path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # up
            d4 = _get_dist(self.__center_x(i), self.__center_y(j - 1),
                           path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # left
            i, j = _next_place(i, j, case, d1, d2, d3, d4)
            path.i_start, path.j_start = i, j

        map_[i, j] = self.__assign_value(i, j, path)
        return map_

    def __matrixfy_one_device(self, df):
        '''
        Parameters:
            - X: the raw DataFrame of only one device

        Returns: the numpy 2d array or sparse matrix, or equivalent Data Structure.
        '''

        map_ = np.zeros(self.resolution)

        for ix, arr in enumerate(df.values):
            sX = arr[x_entry_ix]
            sY = arr[y_entry_ix]
            start_time = arr[time_entry_ix]
            end_time = arr[time_exit_ix]

            i_start, j_start = self.__xy_to_ij(sX, sY)

            if ix == df.shape[0] - 1:
                path = Path(i_start, j_start, i_start, j_start, sX,
                            sY, sX, sY, start_time, end_time)
                map_[path.i_start, path.j_start] = self.__assign_value(
                    path.i_start, path.j_start, path)

            else:
                eX = arr[x_exit_ix]
                eY = arr[y_exit_ix]
                i_end, j_end = self.__xy_to_ij(eX, eY)
                case = _position_case(sX, sY, eX, eY)
                path = Path(i_start, j_start, i_end, j_end, sX,
                            sY, eX, eY, start_time, end_time)
                map_ = self.__matrix_path(map_, path, case)

        return map_
