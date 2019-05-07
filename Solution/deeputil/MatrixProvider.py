import os
import pandas as pd
from Solution.util.BaseUtil import Raw_DF_Reader, time_delta
from scipy import sparse
from Solution.deeputil.Matrixfy import MatrixfyTransformer
from Solution.util.PathFilling import FillPathTransformer


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
        self.__filepath = self.__get_filepath()
        self.pixel = pixel
        self.fill_path = fill_path
        self.value_func = value_func

        r = Raw_DF_Reader()
        self.train = r.train
        self.test = r.test

    def __get_filepath(self):
        dir_ = r"Tmp"
        if self.is_train:
            fname = "train_sparse_matrix.npz"
        else:
            fname = "test_sparse_matrix.npz"
        return os.path.join(dir_, fname)

    def get_sparse_matrix(self):
        '''
            Return: The sparse matrix that contains all the maps
        '''
        if os.path.exists(self.__filepath) and not self.overwrite:
            print("Detected existed required file.")
            self.sparse_matrix = sparse.load_npz(self.__filepath)
            m = Raw_M_Reader(self)
            self.resolution = m.resolution
            if self.is_train:
                self.df_index = m.train_index
            else:
                self.df_index = m.test_index

        else:
            print(
                "No existed required file" if not self.overwrite else "Forced overwrite")
            m = Raw_M_Reader(self)
            if self.is_train:
                self.big_matrix = m.big_train
                self.df_index = m.train_index
            else:
                self.big_matrix = m.big_test
                self.df_index = m.test_index
            self.sparse_matrix = sparse.csr_matrix(self.big_matrix)
            self.__write_matrix()
            self.resolution = m.resolution

        print("Sparse matrix Provided.")
        return self.sparse_matrix

    def __write_matrix(self):
        with open(self.__filepath, "w", encoding="utf-8") as f:
            sparse.save_npz(self.__filepath, self.sparse_matrix)

    def provide_matrix_df(self):
        '''
            Return the required dataframe in CNN.py
        '''
        normal_matrix = self.get_sparse_matrix().todense()
        df = pd.DataFrame(self.df_index)
        tmp_list = []
        for i in range(1, int(normal_matrix.shape[1]/self.resolution[1])):
            tmp_list.append(
                normal_matrix[:, i*self.resolution[1]:(i+1)*self.resolution[1]])
        df['map_'] = tmp_list

        return df


class Raw_M_Reader(object):
    def __init__(self, matrix_provider):
        r = Raw_DF_Reader()
        self.train = r.train
        self.test = r.test

        print("DataFrame read.")

        if matrix_provider.fill_path:
            self.train = FillPathTransformer().transform(self.train)
            self.test = FillPathTransformer().transform(self.test)
            print("Path filled.")

        self.t = MatrixfyTransformer(
            matrix_provider.pixel, matrix_provider.value_func)
        self.t.fit(self.train, self.test)
        self.resolution = self.t.resolution
        self.big_train, self.train_index = self.t.to_matrix_provider(
            self.train)
        self.big_test, self.test_index = self.t.to_matrix_provider(self.test)
