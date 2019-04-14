import datetime
from sklearn.base import BaseEstimator, TransformerMixin



class PositionInMutualTimeExtractor(TransformerMixin, BaseEstimator):
    '''
    Failure Expalanation at the end of fit(self, X).
    '''
    '''
        Select the mutual time span shared by all devices,
        divide the span into a series of time points averagely,
        and estimate the position of each device on these points.
        Assume that devices are moving straightly and at a constant speed.

        Parameters:
            time_points: The divided number of time points

        WARNING:
            When transforming the train_set, it's a must to firstly fit(test_set) and then
            fit_transform(train_set), otherwise the mutual time span may only fit the train
            dataset, leaving some time points without any record in the test set.
            When transforming the test_set, it's the vise versa.
    '''

    def __init__(self, time_points):
        self.time_points = time_points
        self.start = datetime.datetime.strptime("00:00:00", r"%H:%M:%S")
        self.end = datetime.datetime.strptime("23:59:59", r"%H:%M:%S")

    def fit(self, X):
        '''
            Select the mutual time span shared by all devices.
            It has "memory", which means that different dataframes input into fit()
            will be ALL memorized, their mutual time span will be used.

            Parameters:
                X: DataFrame containing column "hash", "time_entry"
        '''
        def first_time_in_a_device(group):
            return group.time_entry.iloc[0]

        def last_time_in_a_device(group):
            return group.time_entry.iloc[-1]

        groups = X.groupby("hash")
        self.start = max(max(groups.apply(first_time_in_a_device)), self.start)
        self.end = min(min(groups.apply(last_time_in_a_device)), self.end)

        '''
            Turned out that there is no mutual time span. Final start > end.
        '''
        return self

    def transform(self, X):
        pass
