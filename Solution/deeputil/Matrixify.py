from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
import math
from datetime import datetime

class Path(object):
    '''
        Contains time and location information of a path
    '''
    def __init__(self,i_start, j_start, i_end, j_end, sPoint_x, sPoint_y, ePoint_x, ePoint_y, start_time, end_time):
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



def get_dist(point_x, point_y, line_x1, line_y1, line_x2, line_y2):
    '''
        Parameters:
            point_x: x coordination of the point
            point_y: y coordination of the point
            line_x1: x coordination of the start point of the line
            line_y1: y coordination of the start point of the line
            line_x2: x coordination of the end point of the line
            line_y2: y coordination of the end point of the line
        Return:
             The distance of the point to the line
    '''
    a = line_y2 - line_y1
    b = line_x1 - line_x2
    c = line_x2 * line_y1 - line_x1 * line_y2
    dis = (math.fabs(a*point_x+b*point_y+c))/(math.pow(a*a+b*b, 0.5))
    return dis

def point_dist(x1, y1, x2, y2):
    '''
        Parameters:
            x1: the x coordination of point1
            y1: the y coordination of point1
            x2: the x coordination of point2
            y2: the y coordination of point2
        Return:
             The distance from one point to the other
    '''
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

def value_function(seconds_0, seconds_1, ratio):
    '''
        The value assignment function
        Parameters haven't been determined.
        It's still a temporary function.
        Return:
            The value to be assigned to the square
    '''
    return (seconds_1 * ratio + seconds_0)/61200




def position_case(sPoint_x, sPoint_y, ePoint_x, ePoint_y):
    '''
        Determine which kind of relative position the 2 point is in
        Parameters:
            path.sPoint_x: the x coordination of the start point
            sPoint_y: the y coordination of the start point
            ePoint_x: the x coordination of the end point
            path.ePoint_y: the y coordination of the end point
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

def next_place(i, j, case, d1, d2, d3, d4):
    '''
        Select next square according to the distance
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
    def __init__(self, pixel):
        self.pixel = pixel

    def fit(self, train, test):

        self.min_x = min(train.x_entry.min(), train.x_exit.min(),
                         test.x_entry.min(), test.x_exit.min())
        self.max_x = max(train.x_entry.max(), train.x_exit.max(),
                         test.x_entry.max(), test.x_exit.max())

        self.min_y = min(train.y_entry.min(), train.y_exit.min(),
                         test.y_entry.min(), test.y_exit.min())
        self.max_y = max(train.y_entry.max(), train.y_exit.max(),
                         test.y_entry.max(), test.y_exit.max())

        return self

    def transform(self, X):
        return pd.DataFrame(X.groupby("hash").apply(self.matrixfy_one_device), columns=["map"])

    def center_x(self, i):
        return (i + 0.5) * self.pixel + self.min_x

    def center_y(self, j):
        return (j + 0.5) * self.pixel + self.min_y

    def xy_to_ij(self, point_x, point_y):
        '''
            Determine which square the point is in
            Parameters:
                point_x: the x coordination of the point
                point_y: the y coordination of the point
                pixel: the size of one square
            Returns:
                The position of the point in the matrix. (like (i, j))
        '''
        return int((point_x - self.min_x) / self.pixel), int((point_y - self.min_y) / self.pixel)

    def assign_value(self, i, j, path):
        '''
            Assign value to the selected square
            Return:
                The value to be assigned to the selected square
        '''
        start_dist = point_dist(self.center_x(
            i), self.center_y(j), path.sPoint_x, path.sPoint_y)
        end_dist = point_dist(self.center_x(
            i), self.center_y(j), path.ePoint_x, path.ePoint_y)
        ratio = start_dist / (start_dist + end_dist)
        base_time = datetime.strptime('1900-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        base_delta = path.start_time - base_time
        delta = path.end_time - path.start_time
        value_number = value_function(base_delta.seconds, delta.seconds, ratio)
        return value_number

    def matrix_path(self, map, path, case):
        '''
            The main function to construct the matrix
            Return:
                The completed matrix path
                The queue that contains information of row, column and value
        '''
        i, j = path.i_start, path.j_start
        while (not ((i == path.i_end) and (j == path.j_end))):
            i, j = path.i_start, path.j_start
            map[i, j] = self.assign_value(i, j, path)
            d1 = get_dist(self.center_x(i + 1), self.center_y(j),
                          path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # down
            d2 = get_dist(self.center_x(i), self.center_y(j + 1),
                          path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # right
            d3 = get_dist(self.center_x(i - 1), self.center_y(j),
                          path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # up
            d4 = get_dist(self.center_x(i), self.center_y(j - 1),
                          path.sPoint_x, path.sPoint_y, path.ePoint_x, path.ePoint_y)  # left
            i, j = next_place(i, j, case, d1, d2, d3, d4)
            path.i_start, path.j_start = i, j

        map[i, j] = self.assign_value(i, j, path)
        return map

    def matrixfy_one_device(self, df):
        '''
        Modify this function only.

        Parameters:
            - X: the raw DataFrame of only one device

        Returns: the numpy 2d array or sparse matrix, or equivalent Data Structure.
        '''
        map = np.zeros(
            (
                math.floor((self.max_x - self.min_x)/self.pixel) + 1,
                math.floor((self.max_y - self.min_y)/self.pixel) + 1
            )
        )
        for i in range(len(df)):
            sX = df.iloc[i, 8]
            sY = df.iloc[i, 9]
            eX = df.iloc[i, 10]
            eY = df.iloc[i, 11]
            start_time = pd.to_datetime(df.iloc[i, 3])
            end_time = pd.to_datetime(df.iloc[i, 4])
            i_start, j_start = self.xy_to_ij(sX, sY)
            i_end, j_end = self.xy_to_ij(eX, eY)
            case = position_case(sX, sY, eX, eY)
            path = Path(i_start, j_start, i_end, j_end, sX, sY, eX, eY, start_time, end_time)
            map = self.matrix_path(map, path, case)
        return map




