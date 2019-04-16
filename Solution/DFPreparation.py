import os
from collections import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from util.Labelling import Labeller
from util.NaiveFeature import (CoordinateInfoExtractor, DistanceInfoExtractor,
                               PathInfoExtractor)
from util.PathFilling import FillPathTransformer
from util.utilFunc import Raw_DF_Reader


class DFProvider(object):

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
        dir_ = r"Tmp"
        fname = self.set_.upper() + "-" + "-".join(self.features) + \
            ("-pathfilled" if self.path_filled else "") + ".csv"
        return os.path.join(dir_, fname)

    def get_df(self):
        if os.path.exists(self.__filepath) and not self.overwrite:
            print("Detected existed required file.")
            with open(self.__filepath, "r", encoding="utf-8") as f:
                self.df = pd.read_csv(f)
        else:
            print(
                "No existed required file" if not self.overwrite else "Forced overwrite"+", recalculating")
            self.df = self.__provide_df()
            self.__write_df()
            print("Newly calculated dataframe retrieved and saved.")

        print("DataFrame Provided.")
        return self.df

    def __write_df(self):
        with open(self.__filepath, "w", encoding="utf-8") as f:
            self.df.to_csv(f, line_terminator="\n")


if __name__ == "__main__":
    for i in ["train", "test"]:
        for j in [True, False]:
            try:
                DFProvider(i, path_filled=j, overwrite=False).get_df()
            except:
                pass
