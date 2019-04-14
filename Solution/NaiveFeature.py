import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from util import Raw_DF_Reader, distance_to_border
from PathFilling import FillPathTransformer


class NaiveDistanceExtractor(TransformerMixin, BaseEstimator):
    '''
        Extract the max, min, average level of the distance of all the points recorded by a device.

        Parameters:
            path_filled: whether the input dataframe is processed by PathFilling.FillPathTransformer
    '''

    def __init__(self, path_filled=True):
        self.path_filled = path_filled

    def fit(self, X):
        return self

    def transform(self, X):
        '''
            Parameters:
                X: Dataframe containing column "hash", "x_entry", "y_entry", "x_exit", "y_exit"

            Returns:
                A Dataframe containing numbers of "hash" rows, three columns.
                The index is the hash value of the device.
                Each column is correspondingly max, min, average distance of all the points recorded by a device.

            If the Extractor is path_filled, it only consider the entries.
            Otherwise it consider both exits and entries.
        '''
        if self.path_filled:
            distance_info_in_group = self.__filled_distance_info_in_group
        else:
            distance_info_in_group = self.__not_filled_distance_info_in_group

        return X.groupby("hash").apply(distance_info_in_group)

    def __filled_distance_info_in_group(self, group):
        '''
            The calculated points are only entries.
        '''
        distance = distance_to_border(group.x_entry, group.y_entry)
        return pd.Series({
            "max_distance": max(distance),
            "min_distance": min(distance),
            "avg_distance": distance.mean()
        })

    def __not_filled_distance_info_in_group(self, group):
        '''
            The calculated points are both entries and exits.
        '''
        distance_1 = distance_to_border(group.x_entry, group.y_entry)
        distance_2 = distance_to_border(
            group.iloc[:-1].x_exit, group.iloc[:-1].y_exit)
        distance = pd.concat([distance_1, distance_2])
        return pd.Series({
            "max_distance": max(distance),
            "min_distance": min(distance),
            "avg_distance": distance.mean()
        })
