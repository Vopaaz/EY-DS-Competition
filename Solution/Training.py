from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import VotingClassifier

from Coordination import BaseTrainExecutor

import logging
from initLogging import init_logging
from params import random_forest_2, gradient_boosting_2

logger = init_logging()
SCORING = make_scorer(f1_score)


class RandomForestExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = random_forest_2
        rand_forest = RandomForestClassifier()
        g_search = GridSearchCV(rand_forest, param_grid,
                                cv=5, scoring=SCORING)
        g_search.fit(feature, target)
        logger.info("Random Forest "+str(g_search.best_params_))
        return g_search.best_estimator_


class GradientBoostingExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = gradient_boosting_2
        g_boosting = GradientBoostingClassifier()
        g_search = GridSearchCV(g_boosting, param_grid, cv=5, scoring=SCORING)
        g_search.fit(feature, target),
        logger.info("GradientBoosting "+str(g_search.best_params_))
        return g_search.best_estimator_


class CombinedExecutor(BaseTrainExecutor):
    def fit(X):
        pass
