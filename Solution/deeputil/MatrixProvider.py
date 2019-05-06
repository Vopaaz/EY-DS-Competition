import os
import pandas as pd
from Solution.util.BaseUtil import Raw_DF_Reader, time_delta
from scipy import sparse
from Solution.deeputil.Matrixfy import MatrixfyTransformer


def naive_value(timestamp):
    start = pd.Timestamp("1900-01-01 00:00:00")
    end = pd.Timestamp("1900-01-01 23:59:59")
    return time_delta(timestamp, start) / time_delta(start, end)

class MProvider(object):
    def __init__(self, big_matrix, df_index, resolution, overwrite=False, is_train=True):
        self.overwrite = overwrite
        self.is_train = is_train
        self.__filepath = self.__get_filepath()
        self.resolution = resolution
        self.big_matrix = big_matrix
        self.df_index = df_index


    def __get_filepath(self):
        dir_ = r"Tmp"
        if self.is_train:
            fname = "train_sparse_matrix.npz"
        else:
            fname = "test_sparse_matrix.npz"
        return os.path.join(dir_, fname)


    def get_sparse_matrix(self):
        if os.path.exists(self.__filepath) and not self.overwrite:
            print("Detected existed required file.")
            self.sparse_matrix = sparse.load_npz(self.__filepath)
        else:
            print(
                "No existed required file" if not self.overwrite else "Forced overwrite")
            self.sparse_matrix = sparse.csr_matrix(self.big_matrix)
            self.__write_matrix()
            print("Newly calculated dataframe retrieved and saved.")

        print("DataFrame Provided.")
        return self.sparse_matrix

    def __write_matrix(self):
        with open(self.__filepath, "w", encoding="utf-8") as f:
            sparse.save_npz(self.__filepath, self.sparse_matrix)

    def provide_matrix_df(self):
        normal_matrix = self.get_sparse_matrix().todense()
        df = pd.DataFrame(self.df_index)
        tmp_list = []
        for i in range(1,int(normal_matrix.shape[1]/self.resolution[1])):
            tmp_list.append(normal_matrix[:,i*self.resolution[1]:(i+1)*self.resolution[1]])
        df['map_'] = tmp_list

        return df





class Raw_M_Reader(object):
    def __init__(self, pixel=1000, value_func=naive_value):
        r = Raw_DF_Reader()
        train = r.train
        test = r.test

        t = MatrixfyTransformer(pixel, value_func)
        t.fit(train, test)
        self.resolution = t.resolution
        self.big_train, self.train_index = t.to_matrix_provider(train)
        self.big_test, self.test_index = t.to_matrix_provider(test)


if __name__ == '__main__':
    '''
        Test
    '''
    m = Raw_M_Reader()
    sp = MProvider(m.big_train, m.train_index, m.resolution)
    df = sp.provide_matrix_df()
    print(df)