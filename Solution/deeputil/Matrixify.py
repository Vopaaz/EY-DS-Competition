from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


def test_time_value_func(timestamp):
    start = pd.Timestamp("1900-01-01 00:00:00")
    return (timestamp - start).total_seconds()


class MatrixfyTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, resolution, time_value_func):
        self.resolution = resolution
        self.time_value_func = time_value_func

    def fit(self, train, test):

        self.min_x = min(train.x_entry.min(), train.x_exit.min(),
                         test.x_entry.min(), test.x_exit.min())
        self.max_x = max(train.x_entry.max(), train.x_exit.max(),
                         test.x_entry.max(), test.x_exit.max())

        self.min_y = min(train.y_entry.min(), train.y_exit.min(),
                         test.y_entry.min(), test.y_exit.min())
        self.max_y = max(train.y_entry.max(), train.y_exit.max(),
                         test.y_entry.max(), test.y_exit.max())

        self.__precalculation()
        return self

    def __precalculation(self):
        self.pixel_width = (self.max_x - self.min_x)/self.resolution[1]
        self.pixel_height = (self.max_y - self.min_y)/self.resolution[0]
        self.pixel_min_border = min(self.pixel_width, self.pixel_height)
        self.pixel_diagonal = (self.pixel_width ** 2 +
                               self.pixel_height ** 2) ** 0.5

    def transform(self, X):
        return pd.DataFrame(X.groupby("hash").apply(self.__matrixfy_one_device), columns=["map"])

    def __matrixfy_one_device(self, df):
        '''
        Modify this function only.

        Parameters:
            - X: the raw DataFrame of only one device

        Returns: the numpy 2d array or sparse matrix, or equivalent Data Structure.
        '''
        map_ = np.zeros(self.resolution)
        return map_



if __name__ == "__main__":
    import sys
    sys.path.append("Solution")
    sys.path.append(".")
    from Solution.util.utilFunc import Raw_DF_Reader
    r = Raw_DF_Reader()
    print(MatrixfyTransformer((10, 10), test_time_value_func).fit(
        r.test.iloc[:100], r.train[:100]).transform(r.train[:100]))
