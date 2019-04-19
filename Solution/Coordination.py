import pandas as pd


class NanCoordiantor(object):
    r'''
        The Coordinator to handle nan values in the train/test set and apply different strategies.

        Parameters:
            - train: The train DataFrame
            - test: The test DataFrame
            - strategy: ["fill_0"/"separate_all"/"separate_part"], str
                - "fill_0": fill all null values with 0
                - "separate_all" / "separate_part": see the 'Explanation of Separate Strategy' part

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

        WARNING: To apply these two strategies, it is required that the train and test set has
        'similar null value structure'. For example, in the previous case, a test record with null A, C
        and non-null B is NOT ALLOWED.
    '''

    def __init__(self, train, test, strategy="fill_0"):
        if strategy not in ["fill_0", "separate_all", "separate_part"]:
            raise ValueError(
                "Parameter strategy must be 'fill_0',\
                     'separate_all', or 'separate_part', now it's {}.".format(strategy))

        self.strategy = strategy
        self.trains = train
        self.tests = test

        STRATEGIES = {
            "fill_0": self.__fill_0,
            "separate_all": self.__separate_all,
            "separate_part": self.__separate_part
        }
        STRATEGIES[strategy]()

    def __fill_0(self):
        self.trains = [self.trains.fillna(0)]
        self.tests = [self.tests.fillna(0)]

    def __separate_all(self):
        pass

    def __separate_part(self):
        pass

    def fit(self, TrainExecutor, *params, **kwparams):
        '''
            Fit machine learning models based on the selected strategy.

            Parameters:
                - TrainExecutor: a class that handles the machine learning routines from preprocessing to training.
                  It must provide the following APIs:
                    - fit: takes one train set as the parameter and return a model
                - params & kwparams: the parameters to be passed to the TrainExecutor when initializing the object.

            Returns:
                - A list of trained models.
        '''

        self.models = [TrainExecutor(*params, **kwparams).fit(i)
                       for i in self.trains]
        return self.models

    def predict(self, PredictExecutor, *params, **kwparams):
        res = [PredictExecutor(model, *params, **kwparams).predict(test)
               for test, model in zip(self.tests, self.models)]
        return pd.concat(res, axis=0)


class BaseTrainExecutor(object):
    def fit(self, train):
        raise NotImplementedError


class BasePredictExecutor(object):
    def __init__(self, model, *args, **kwargs):
        self.model = model

    def predict(self, test):
        raise NotImplementedError
