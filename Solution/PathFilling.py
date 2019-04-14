import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from util import Raw_DF_Reader

traj_id_ix = 1
time_entry_ix = 2
time_exit_ix = 3
x_entry_ix = 7
y_entry_ix = 8
x_exit_ix = 9
y_exit_ix = 10


class FillPathTransformer(TransformerMixin, BaseEstimator):
    '''
        Fill the disconnected paths for raw train or test dataframe.
        When it is applied to the train dataframe, the test dataframe
        MUST be applied as well and vise versa, otherwise the prediction
        result will fail.
    '''

    def fit(self, X):
        return self

    def transform(self, X):
        '''
            Parameter:
                X is raw train or test dataframe.
            Return:
                The dataframe that all the paths are connected and inserted.
        '''
        return pd.concat([self.__connect_one_device(i) for i in X.groupby("hash")], axis=0).reset_index(drop=True)

    def __connect_one_device(self, group):
        '''
            Connect all paths of a device.
            Parameter:
                group: a pandas Groupby object, group by "hash"
            Return:
                A dataframe that all the paths of this device are connected and inserted.
        '''
        hash_, df = group[0], group[1]

        def add_p5_suffix(x):
            return x + ".5"

        new_df = pd.DataFrame([[
            hash_,  # hash
            add_p5_suffix(df.iloc[i, traj_id_ix]),  # traj_id
            df.iloc[i, time_exit_ix],  # time_entry
            df.iloc[i+1, time_entry_ix],    # time_exit
            np.nan,  # vmax
            np.nan,  # vmin
            np.nan,  # vmean
            df.iloc[i, x_exit_ix],  # x_entry
            df.iloc[i, y_exit_ix],  # y_entry
            df.iloc[i+1, x_entry_ix],    # x_exit
            df.iloc[i+1, y_entry_ix]     # y_exit
        ] for i in range(df.shape[0] - 1)], columns=df.columns)

        return pd.concat([df, new_df], axis=0).sort_values(by="time_entry", ascending=True)
