import pandas as pd
from util import Raw_DF_Reader
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class FillPathTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X):
        return self

    def transform(self, X):
        return pd.concat([self.__connect_one_device(i) for i in X.groupby("hash")], axis=0).reset_index(drop=True)

    def __connect_one_device(self, group):
        hash_, df = group[0], group[1]

        def rename_traj_id(x):
            return x + ".5"

        new_df = pd.DataFrame([[
            hash_,
            rename_traj_id(df.iloc[i, 1]),
            df.iloc[i, 3],
            df.iloc[i+1, 2],
            np.nan,
            np.nan,
            np.nan,
            df.iloc[i, 9],
            df.iloc[i, 10],
            df.iloc[i+1, 7],
            df.iloc[i+1, 8]
        ] for i in range(df.shape[0] - 1)], columns=df.columns)

        return pd.concat([df,new_df], axis=0).sort_values(by="time_entry", ascending=True)


if __name__ == "__main__":
    df = Raw_DF_Reader().test
    print(FillPathTransformer().fit_transform(df))
