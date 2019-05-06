import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from Solution.util.BaseUtil import (Raw_DF_Reader, distance_between,
                                    distance_to_border, time_delta)


class DistanceInfoExtractor(TransformerMixin, BaseEstimator):
    '''
        Features extracted:
            - The max, min, average level of the distance of all the points recorded by a device.
            - The difference between the distance of the entry of the first path and the exit of the last one.
            - The difference between the distance of the entry and the exit of the last path.

        Parameters:
            path_filled: whether the input dataframe is processed by PathFilling.FillPathTransformer

        All the distances will be l1 distance.
    '''

    def __init__(self, path_filled=False, *args, **kwargs):
        self.path_filled = path_filled

    def fit(self, X):
        return self

    def transform(self, X):
        '''
            Parameters:
                X: Dataframe containing column "hash", "x_entry", "y_entry", "x_exit", "y_exit"

            Returns:
                A Dataframe containing numbers of "hash" rows, five columns.
                The index is the hash value of the device.
                Each column is a feature, as described by the class docstring.

            The last path will not be considered.
        '''

        if self.path_filled:
            distance_info_in_group = self.__filled_distance_info_in_group
        else:
            distance_info_in_group = self.__not_filled_distance_info_in_group

        return X.groupby("hash").apply(distance_info_in_group)

    def __filled_distance_info_in_group(self, group):
        '''
            Extract the features from the records of one device.
            The result will be identical to that of using self.__not_filled_distance_info_in_group()
            But the performance might be slightly better.
        '''

        distance = distance_to_border(group.x_entry, group.y_entry)

        if distance.empty:
            return pd.Series({
                "max_distance": np.nan,
                "min_distance": np.nan,
                "avg_distance": np.nan,
                "start_end_dist_diff": np.nan,
                "last_path_dist_diff": np.nan
            })

        try:
            return pd.Series({
                "max_distance": distance.max(),
                "min_distance": distance.min(),
                "avg_distance": distance.mean(),
                "start_end_dist_diff": distance.iloc[-1] - distance.iloc[0],
                "last_path_dist_diff": distance.iloc[-1] - distance.iloc[-2]
            })
        except:
            return pd.Series({
                "max_distance": distance.max(),
                "min_distance": distance.min(),
                "avg_distance": distance.mean(),
                "start_end_dist_diff": distance.iloc[-1] - distance.iloc[0],
                "last_path_dist_diff": np.nan
            })

    def __not_filled_distance_info_in_group(self, group):
        '''
            Extract the features from the records of one device.
        '''
        group_considered = group.iloc[:-1]

        distance_1 = distance_to_border(
            group_considered.x_entry, group_considered.y_entry)
        distance_2 = distance_to_border(
            group_considered.x_exit, group_considered.y_exit)

        distance = pd.concat([distance_1, distance_2])

        if distance.empty:
            return pd.Series({
                "max_distance": np.nan,
                "min_distance": np.nan,
                "avg_distance": np.nan,
                "start_end_dist_diff": np.nan,
                "last_path_dist_diff": np.nan
            })
        else:
            return pd.Series({
                "max_distance": distance.max(),
                "min_distance": distance.min(),
                "avg_distance": distance.mean(),
                "start_end_dist_diff": distance.iloc[-1] - distance.iloc[0],
                "last_path_dist_diff": distance.iloc[-1] - distance.iloc[group.shape[0]-2]
            })


class PathInfoExtractor(TransformerMixin, BaseEstimator):
    '''
        Features extracted:
            - The min, max, average level of the length of all the paths recorded by a device
            - The min, max, average level of the average velocity of all the paths recorded by a device
    '''

    def __init__(self, *args, **kwargs):
        super().__init__()

    def fit(self, X):
        return self

    def transform(self, X):
        '''
            Parameters:
                X: Dataframe containing column:
                "hash", "x_entry", "y_entry", "x_exit", "y_exit", "time_entry", "time_exit"

            Returns:
                A Dataframe containing numbers of "hash" rows, six columns.
                The index is the hash value of the device.
                Each column is a feature, as described by the class docstring.

            The last path will not be considered.
        '''

        return X.groupby("hash").apply(self.__path_info_in_group)

    def __path_info_in_group(self, group):
        '''
            Extract the features from the records of one device.
        '''

        group_considered = group.iloc[:-1]
        lengths = distance_between(group_considered.x_entry, group_considered.y_entry,
                                   group_considered.x_exit, group_considered.y_exit)

        time_deltas = time_delta(
            group_considered.time_entry, group_considered.time_exit)

        velocities = pd.concat([lengths, time_deltas], axis=1).apply(
            lambda series: series.iloc[0]/series.iloc[1] if series.iloc[1] != 0 else np.nan, axis=1)

        if velocities.empty:
            velocities = pd.Series([])

        return pd.Series({
            "max_length": lengths.max(),
            "min_length": lengths.min(),
            "avg_length": lengths.mean(),
            "max_velocity": velocities.max(),
            "min_velocity": velocities.min(),
            "avg_velocity": velocities.mean()
        })


class CoordinateInfoExtractor(TransformerMixin, BaseEstimator):
    '''
        Features Extracted:
            - The coordinate of the start point of the last path (the unknown, to be predicted path).
    '''

    def __init__(self, *args, **kwargs):
        super().__init__()

    def fit(self, X):
        return self

    def transform(self, X):
        '''
            Parameters:
                X: Dataframe containing column:
                "hash", "x_entry", "y_entry"

            Returns:
                A Dataframe containing numbers of "hash" rows, two columns.
                The index is the hash value of the device.
                Each column is a feature, as described by the class docstring.
        '''

        return X.groupby("hash").apply(
            lambda group: group[["x_entry", "y_entry"]].iloc[-1]
        ).rename(columns={
            "x_entry": "x_last_point",
            "y_entry": "y_last_point"
        })
