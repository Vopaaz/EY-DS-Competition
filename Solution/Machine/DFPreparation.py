import os
from collections import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from Solution.util.BaseUtil import Raw_DF_Reader
from Solution.util.Labelling import Labeller
from Solution.util.NaiveFeature import (CoordinateInfoExtractor,
                                        DistanceInfoExtractor,
                                        PathInfoExtractor)
from Solution.util.PathFilling import FillPathTransformer


class DFProvider(object):
    '''
        Provide the applicable DataFrame, each row is one device,
        each column is one feature (or the prediction label if for train set).
        Invalid values (np.nan) are contained.

        The calculate result will be automately saved in Tmp/,
        the filename is also auto-mapped.

        Parameters:
            - set_: ["train"/"test"], get train set or test set.
            - features: str "all" or list containing the following options:
                - "coordinate"
                - "distance"
                - "path"
                Each means a set of features. Default "all".
            - path_filled: boolean, whether or not to apply the path filling routine. Default True
            - overwrite: boolean, whether or not ignore the existed csv file and recalculate the dataframe.

        ---
        Inserting New Extractors by modifying the code:
        1. add the extractor class to ALL_EXTRACTORS constant dictionary.
        2. (optional) pass the parameters to the extractors to the __initialize_extractors() function.
    '''
    ALL_EXTRACTORS = {
        "coordinate": CoordinateInfoExtractor,
        "distance": DistanceInfoExtractor,
        "path": PathInfoExtractor
    }
    ALL_FEATURES = ALL_EXTRACTORS.keys()

    def __init__(self, set_, features="all", path_filled=True, overwrite=False):

        if set_ in ["train", "test"]:
            self.set_ = set_
        else:
            raise ValueError(
                "Parameter 'set_' can only be 'train' or 'test', now it is {}.".format(set_))

        self.overwrite = overwrite
        self.path_filled = path_filled

        feature_value_error = ValueError(
            "Parameter 'features' (or its elements) can only be in {}, the parameter given is {}".format(
                self.ALL_FEATURES, features)
        )

        if features == "all":
            self.extractors = self.ALL_EXTRACTORS.values()
            self.features = self.ALL_FEATURES
        elif isinstance(features, Iterable):
            self.extractors = []
            self.features = list(set(features))
            for i in set(features):
                if i in self.ALL_FEATURES:
                    self.extractors.append(self.ALL_EXTRACTORS[i])
                else:
                    raise feature_value_error
        else:
            raise feature_value_error

        self.__filepath = self.__get_filepath()

    def __provide_df(self):
        '''
            Returns the required DataFrame.
        '''
        self.__initialize_extractors()
        if self.set_ == "train":
            self.extractor_objs.append(Labeller())
            raw_df = Raw_DF_Reader().train
        else:
            raw_df = Raw_DF_Reader().test

        if self.path_filled:
            print("Filling paths")
            raw_df = FillPathTransformer().fit_transform(raw_df)
            print("Path-filling finished.")

        dfs = []
        for i in self.extractor_objs:
            print("Start: ", i.__class__.__name__)
            dfs.append(i.fit_transform(raw_df))
            print("Finished: ", i.__class__.__name__,)

        return pd.concat(dfs, axis=1)

    def __initialize_extractors(self):
        self.extractor_objs = [i(self.path_filled) for i in self.extractors]

    def __get_filepath(self):
        '''
            Map the parameters of the provider to an identifiable filepath.
        '''
        dir_ = r"Tmp"
        fname = self.set_.upper() + "-" + "-".join(self.features) + \
            ("-pathfilled" if self.path_filled else "") + ".csv"
        return os.path.join(dir_, fname)

    def get_df(self):
        '''
            Returns the required DataFrame.
        '''
        if os.path.exists(self.__filepath) and not self.overwrite:
            print("Detected existed required file.")
            with open(self.__filepath, "r", encoding="utf-8") as f:
                self.df = pd.read_csv(f)
        else:
            print(
                "No existed required file" if not self.overwrite else "Forced overwrite"+", recalculating.")
            self.df = self.__provide_df().apply(pd.to_numeric, errors="coerce")
            self.__write_df()
            print("Newly calculated dataframe retrieved and saved.")

        print("DataFrame Provided.")
        return self.df

    def __write_df(self):
        with open(self.__filepath, "w", encoding="utf-8") as f:
            self.df.to_csv(f, line_terminator="\n")


'''
    The following code can calculate and save the most useful csv files.
'''

if __name__ == "__main__":
    import threading
    for i in ["train", "test"]:
        for j in [True, False]:
            try:
                t = threading.Thread(
                    target=DFProvider(i, path_filled=j, overwrite=True).get_df
                )
                t.start()
            except Exception as e:
                print(e)
