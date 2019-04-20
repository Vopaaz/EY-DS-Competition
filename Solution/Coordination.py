import logging

import pandas as pd
import datetime
import os
from sklearn.preprocessing import StandardScaler

from util.utilFunc import Raw_DF_Reader


def split_hash_feature_target(full_df):
    '''
        Split the hash, feature and target column from a DataFrame.
        Parameter:
            - full_df: the DataFrame to be splitted. It MUST contain column "hash", but "target" is optional.

        Return:
            - hash_: hash Series
            - feature: feature DataFrame
            - target: target Series, or None
    '''
    has_target = "target" in full_df.columns.values

    hash_ = full_df.hash
    feature = full_df.drop(columns=["hash"])
    target = full_df.target if has_target else None

    if has_target:
        feature = feature.drop(columns=["target"])

    return hash_, feature, target


def _check_preprocessed(func):
    '''
        Decorator: Check whether the preprocess is applied.
        For decoration of the methods of NanCoordinator.
    '''

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
        raise NotImplementedError

    def __separate_part(self):
        raise NotImplementedError

    def preprocess(self, PreprocessingExecutor, *args, **kwargs):
        '''
            Apply the same preprocessing process to the train and test sets.

            Parameters:
                - PreprocessingExecutor: a class that handles the preprocessing routines.
                  It must provide the following APIs:
                    - fit: use the dataset to fit the transformer, RETURN ITSELF.
                    - transform: preprocess and transform the dataset, return a FULL DataFrame
                - args & kwargs: the parameters to be passed to the Preprocessing Executor when initializing the object.

            WARNING:
            The Executors should receive the full (containing hash and target) as the parameter,
            and its transform method should also be a DataFrame containing all the columns.
        '''
        self.preprocessors = [
            PreprocessingExecutor(*args, **kwargs).fit(i) for i in self.trains]
        self.trains = [
            preprocessor.transform(i) for (i, preprocessor) in zip(self.trains, self.preprocessors)]
        self.tests = [
            preprocessor.transform(i) for (i, preprocessor) in zip(self.tests, self.preprocessors)]
        self.preprocessed = True

    @_check_preprocessed
    def fit(self, TrainExecutor, *args, **kwargs):
        '''
            Fit machine learning models based on the selected strategy.

            Parameters:
                - TrainExecutor: a class that handles the machine learning training routines.
                  It must provide the following APIs:
                    - fit: takes one train set as the parameter and return a model, who has a 'predict' API.
                - args & kwargs: the parameters to be passed to the TrainExecutor when initializing the object.

            Returns:
                - A list of trained models.

            WARNING:
            The Executors should receive the full (containing hash and target) as the parameter.
        '''

        self.models = [TrainExecutor(*args, **kwargs).fit(i)
                       for i in self.trains]
        return self.models

    @_check_preprocessed
    def predict(self):
        '''
            Predict the results based on the fitted models.
            The result of the splitted groups will be combined in one DataFrame.

            Returns: a full DataFrame containing columns "hash" and "target".
        '''
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
    '''
        The Executor Base class.
    '''

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def split_hash_feature_target(self, X):
        '''
            Wrap the split_hash_feature_target method (at the beginning of this file).
        '''
        return split_hash_feature_target(X)

    def combine_hash_feature_target(self, hash_, feature, target, feature_cols=None):
        '''
        Combine the hash, feature and target column into a DataFrame.

        Parameter:
            - hash_: hash Series
            - feature: feature np.ndarray (or pd.DataFrame)
            - target: target Series, or None
            - feature_cols: optional, the column name for the features.

        Return:
            - full_df: the combined DataFrame
        '''
        if not feature_cols:
            feature_cols = pd.RangeIndex(0, feature.shape[1])

        res = pd.DataFrame(feature, columns=feature_cols)
        res.insert(0, "hash", hash_)

        if isinstance(target, pd.Series):
            res["target"] = target
        return res


class BasePreprocessingExecutor(BaseExecutor):
    '''
        Base class for the preprocessing executors.
    '''

    def fit(self, train):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class BaseTrainExecutor(BaseExecutor):
    '''
        Base class for the train executors
    '''

    def fit(self, train):
        raise NotImplementedError


class Submitter(object):
    '''
        Convert the hash_target DataFrame to the (traj_)id_target DataFrame,
        which is the required format.
        It can also save the result and with some memo infomation.

        Parameters:
            - hash_result: the prediction result by other components.
                Should have two columns: "hash" and "target"

        Attributes:
            - result: the transformed DataFrame with columns "id" and "target"
    '''

    def __init__(self, hash_result):
        self.hash_result = hash_result
        self.__transform_result()

    def __transform_result(self):
        '''
            Transform the hash_target DataFrame to
        '''
        raw_test = Raw_DF_Reader().test
        groups = raw_test.groupby("hash")
        result = pd.DataFrame()
        result["id"] = self.hash_result.apply(
            lambda series: groups.get_group(series.hash).trajectory_id.iloc[-1], axis=1)
        result["target"] = self.hash_result.target
        self.result = result

    def save(self, memo=""):
        '''
            Save the result DataFrame to csv file.
            The target diretory is "Result". The file will be named by monthday-hour-minute-second.

            Parameters:
                - memo: A string that describes this result DataFrame, it will be written in the memo.txt under the Result dir. 
        '''
        filename = datetime.datetime.now().strftime(r"%m%d-%H-%M-%S") + ".csv"
        filepath = os.path.join("Result", filename)
        self.result.to_csv(filepath, encoding="utf-8",
                           index=False, line_terminator="\n")

        with open(os.path.join("Result", "memo.txt"), "a+", encoding="utf-8") as f:
            f.write(filename)
            f.write("\t")
            f.write(str(memo))
            f.write("\n")
