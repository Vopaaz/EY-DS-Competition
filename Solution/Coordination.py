import logging

import pandas as pd

from util.utilFunc import Raw_DF_Reader


def split_hash_feature_target(full):

    has_target = "target" in full.columns.values

    hash_ = full.hash
    feature = full.drop(columns=["hash"])
    target = full.target if has_target else None

    if has_target:
        feature = feature.drop(columns=["target"])

    return hash_, feature, target


def _check_preprocessed(func):
    def inner(self, *args, **kwargs):
        if not self.preprocessed:
            msg = "The datasets are NOT PREPROCESSED in the coordinator. " +\
                "Please check that preprocessing routine is executed somewhere in the pipeline."
            logging.warning(msg)
        return func(self, *args, **kwargs)
    return inner


class NanCoordiantor(object):
    r'''
        The Coordinator to handle nan values in the train/test set and apply different strategies.

        Parameters:
            - train: The train DataFrame
            - test: The test DataFrame
            - strategy: ["drop"/"fill_0"/"separate_all"/"separate_part"], str
                - "drop": use only the attributes that are non-null for all records.
                - "fill_0": fill all null values with 0
                - "separate_*": see the 'Explanation of Separate Strategy' part

        Explanation of Separate Strategy:

            Example of train set (v means value and N means nan):
                A   B   C
            0   v   v   v
            1   v   v   v
            2   v   v   N
            3   v   v   N
            4   v   N   N
            5   v   N   N

        separate_all:
            - Use (0-5).A to train the model and predict those whose non-null feature is only A
            - Use (0-3).AB to train the model and predict those whose non-null feature is A, B
            - Use (0-1).ABC to train the model and predict those whose non-null feature is A, B, and C

        separate_part:
            - Use (4-5).A to train the model and predict those whose non-null feature is only A
            - Use (2-3).AB to train the model and predict those whose non-null feature is A, B
            - Use (0-1).ABC to train the model and predict those whose non-null feature is A, B, and C

        WARNING:
        1. To apply "drop", "separate_*" strategies, it is required that the train and test set has
        'similar null value structure'. For example, in the previous case, a test record
        with null A, C and non-null B is NOT ALLOWED.

        2. In the whole process flow, the "Executors" should receive a FULL DataFrame and
        return a FULL DataFrame. By full it means that it should contain the column names,
        including the "hash" and "target" rows in the train set. i.e. The executors, rather
        than the Coornidator should handle the splitting of features and labels, etc.
    '''

    def __init__(self, train, test, strategy="fill_0"):

        self.STRATEGIES = {
            "drop": self.__drop,
            "fill_0": self.__fill_0,
            "separate_all": self.__separate_all,
            "separate_part": self.__separate_part
        }

        if strategy not in self.STRATEGIES:
            raise ValueError(
                "Parameter strategy must be 'fill_0',\
                     'separate_all', or 'separate_part', now it's {}.".format(strategy))

        self.strategy = strategy

        # The variables are named 'trains' and 'tests' rather than their singular form,
        # because then they will be transformed into a list according to their strategies.
        self.trains = train
        self.tests = test

        self.STRATEGIES[strategy]()
        # Now the self.trains and self.tests are lists, in accordance with their plural form.

        self.models = None
        self.preprocessed = False

    def __drop(self):
        self.trains = [self.trains.dropna(axis=1)]
        self.tests = [self.tests.dropna(axis=1)]

    def __fill_0(self):
        self.trains = [self.trains.fillna(0)]
        self.tests = [self.tests.fillna(0)]

    def __separate_all(self):
        pass

    def __separate_part(self):
        pass

    def preprocess(self, PreprocessingExecutor, *params, **kwparams):
        '''
            Apply the same preprocessing process to the train and test sets.

            Parameters:
                - PreprocessingExecutor: a class that handles the preprocessing routines.
                  It must provide the following APIs:
                    - fit: use the dataset to fit the transformer
                    - transform: preprocess and transform the dataset
                - params & kwparams: the parameters to be passed to the Preprocessing Executor when initializing the object.

        '''
        self.preprocessors = [
            PreprocessingExecutor(*params, **kwparams).fit(i) for i in self.trains]
        self.trains = [
            preprocessor.transform(i) for (i, preprocessor) in zip(self.trains, self.preprocessors)]
        self.tests = [
            preprocessor.transform(i) for (i, preprocessor) in zip(self.tests, self.preprocessors)]
        self.preprocessed = True

    @_check_preprocessed
    def fit(self, TrainExecutor, *params, **kwparams):
        '''
            Fit machine learning models based on the selected strategy.

            Parameters:
                - TrainExecutor: a class that handles the machine learning training routines.
                  It must provide the following APIs:
                    - fit: takes one train set as the parameter and return a model
                - params & kwparams: the parameters to be passed to the TrainExecutor when initializing the object.

            Returns:
                - A list of trained models.
        '''

        self.models = [TrainExecutor(*params, **kwparams).fit(i)
                       for i in self.trains]
        return self.models

    @_check_preprocessed
    def predict(self):

        def predict_one_group(test, model):
            hash_, feature, _ = split_hash_feature_target(test)
            return pd.DataFrame({
                "hash": hash_,
                "target": model.predict(feature)
            })

        res = [predict_one_group(test, model)
               for (test, model) in zip(self.tests, self.models)]
        return pd.concat(res, axis=0)


class BaseExecutor(object):
    def split_hash_feature_target(self, X):
        return split_hash_feature_target(X)

    def combine_hash_feature_target(self, hash_, feature, target, feature_cols=None):
        if not feature_cols:
            feature_cols = pd.RangeIndex(0, feature.shape[1])

        res = pd.DataFrame(feature, columns=feature_cols)
        res.insert(0, "hash", hash_)

        if isinstance(target, pd.Series):
            res["target"] = target
        return res


class BasePreprocessingExecutor(BaseExecutor):
    def fit(self, train):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class BaseTrainExecutor(BaseExecutor):
    def fit(self, train):
        raise NotImplementedError


class Submitter(object):
    def __init__(self):
        self.raw_test = Raw_DF_Reader().test

    def transform(self, hash_result):
        groups = self.raw_test.groupby("hash")
        res = pd.DataFrame()
        res["id"] = hash_result.apply(
            lambda series: groups.get_group(series.hash).trajectory_id.iloc[-1], axis=1)
        res["target"] = hash_result.target
        return res
