import os
import math
import pandas as pd
import numpy as np
from Solution.util.BaseUtil import Raw_DF_Reader, time_delta
from scipy import sparse
from Solution.deeputil.Matrixfy import MatrixfyTransformer
from Solution.util.PathFilling import FillPathTransformer



X_RANGE = 36100.91086425679
Y_RANGE = 340258.3224131949
#X_RANGE = 32481.914218568243   # When test range is 30
#Y_RANGE = 293142.28293212503   # When test range is 30

def naive_value(timestamp):
    start = pd.Timestamp("1900-01-01 00:00:00")
    end = pd.Timestamp("1900-01-01 23:59:59")
    return time_delta(timestamp, start) / time_delta(start, end)


class MProvider(object):
    '''
        Provide sparse matrix, normal matrix and the required dataframe
        Parameters:
            - pixel: representing the width and height for one pixel in the map
            - fill_path: boolean, whether or not to let the map be the path-filled version
            - value_func: the value function passed to the MatrixfyTransformer
            - overwrite: boolean, whether or not to overwrite the file
            - is_train: boolean, whether the dataframe is train or test

        Attributes:
            - overwrite
            - is_train
            - __filepath: store the file path
            - pixel
            - fill_path
            - value_func
    '''

    def __init__(self, set_, pixel=1000, fill_path=True, value_func=naive_value, overwrite=False):
        self.overwrite = overwrite
        if set_ == "train":
            self.is_train = True
        elif set_ == "test":
            self.is_train = False
        else:
            raise ValueError(
                "Parameter 'set_' can only be 'train' or 'test'. Now it's {}.".format(set_))

        self.pixel = pixel
        self.fill_path = fill_path
        self.value_func = value_func
        self.__filepath = self.__get_filepath()
        self.__indexpath = self.__get_indexpath()
        self.resolution = self.__get_resolution()


    def __get_indexpath(self):
        dir_ = r"Tmp"
        if self.is_train:
            name = "train_index"
        else:
            name = "test_index"
        if self.fill_path:
            fp = "fill"
        else:
            fp = "nfill"
        fname = name + "-p" + str(self.pixel) + "-" + fp + ".csv"
        return os.path.join(dir_, fname)

    def __get_filepath(self):
        dir_ = r"Tmp"
        if self.is_train:
            name = "train_matrix"
        else:
            name = "test_matrix"
        if self.fill_path:
            fp = "fill"
        else:
            fp = "nfill"
        fname = name + "-p" + str(self.pixel) + "-" + fp + ".npz"
        return os.path.join(dir_, fname)

    def __get_resolution(self):
        return (math.floor(X_RANGE / self.pixel) + 1,
                math.floor(Y_RANGE / self.pixel) + 1)

    def get_sparse_matrix(self):
        '''
            Return: The sparse matrix that contains all the maps
        '''
        if os.path.exists(self.__filepath) and os.path.exists(self.__indexpath) and not self.overwrite:
            print("Detected existed required file.")
            self.sparse_matrix = sparse.load_npz(self.__filepath)
            with open(self.__indexpath, "r", encoding="utf-8") as f1:
                self.df_index = pd.read_csv(f1)


        else:
            print(
                "No existed required file" if not self.overwrite else "Forced overwrite")
            self.provide_matrix_and_index()
            self.sparse_matrix = sparse.csr_matrix(self.big_matrix)
            self.df_index = pd.DataFrame(self.df_index)
            self.__write_matrix()

        print("Sparse matrix Provided.")
        return self.sparse_matrix

    def __write_matrix(self):
        sparse.save_npz(self.__filepath, self.sparse_matrix)
        with open(self.__indexpath, "w", encoding="utf-8") as f1:
            self.df_index.to_csv(f1)

    def provide_matrix_df(self):
        '''
            Return the required dataframe in CNN.py
        '''
        normal_matrix = self.get_sparse_matrix().todense()
        df = self.df_index
        tmp_list = []
        for i in range(0, int(normal_matrix.shape[1]/self.resolution[1])):
            tmp_list.append(
                normal_matrix[:, i*self.resolution[1]:(i+1)*self.resolution[1]])
        df['map_'] = tmp_list
        df = df[['hash', 'map_']]
        df.set_index("hash", inplace=True)

        return df

    def provide_matrix_and_index(self):
        r = Raw_DF_Reader()
        self.train = r.train
        self.test = r.test

        print("DataFrame read.")

        if self.fill_path and self.is_train:
            self.train = FillPathTransformer().transform(self.train)
            print("Path filled.")
        if self.fill_path and not self.is_train:
            self.test = FillPathTransformer().transform(self.test)
            print("Path filled.")

        if self.is_train:
            self.big_matrix, self.df_index = self.matrix_and_index(self.train)
        else:
            self.big_matrix, self.df_index = self.matrix_and_index(self.test)

    def matrix_and_index(self, df):
        '''
            Provide a big normal matrix that contains all the maps in df
            Parameters:
                df: The raw dataframe or pathfilled dataframe
            Returns:
                self.big_matrix: a big normal matrix that contains all the maps in df
                tmp_df.index: hash used for constructing output dataframe in MatrixProvider
        '''
        t= MatrixfyTransformer()
        t.fit(self.train,self.test)
        tmp_df = t.transform(df)['map_']
        self.big_matrix = tmp_df.iloc[0]
        for i in range(1, len(tmp_df)):
            self.big_matrix = np.concatenate([self.big_matrix, tmp_df.iloc[i]], axis=1)

        return self.big_matrix, tmp_df.index


if __name__ == '__main__':
    train_provider = MProvider("train")
    test_provider = MProvider("test")
    train_maps = train_provider.provide_matrix_df()
    test_maps = test_provider.provide_matrix_df()