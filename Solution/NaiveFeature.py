from util import Raw_DF_Reader, distance_to_border
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class NaiveDistanceExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, path_filled=True):
        self.path_filled = path_filled

    def fit(self, X):
        return self

    def transform(self, X):
        if self.path_filled:
            distance_info_in_group = self.__filled_distance_info_in_group
        else:
            distance_info_in_group = self.__not_filled_distance_info_in_group

        return X.groupby("hash").apply(distance_info_in_group).reset_index(drop=True)

    def __filled_distance_info_in_group(self, group):
        distance = distance_to_border(group.x_entry, group.y_entry)
        return pd.Series({
            "max_distance": max(distance),
            "min_distance": min(distance),
            "avg_distance": distance.mean()
        })

    def __not_filled_distance_info_in_group(self, group):
        distance_1 = distance_to_border(group.x_entry, group.y_entry)
        distance_2 = distance_to_border(
            group.iloc[:-1].x_exit, group.iloc[:-1].y_exit)
        distance = pd.concat([distance_1, distance_2])
        return pd.Series({
            "max_distance": max(distance),
            "min_distance": min(distance),
            "avg_distance": distance.mean()
        })
