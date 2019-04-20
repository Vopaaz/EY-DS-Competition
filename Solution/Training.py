from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import VotingClassifier

from Coordination import BaseTrainExecutor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(r"log\trainLog.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

SCORING = make_scorer(f1_score)


class RandomForestExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = {
            "n_estimators": [10, 100, 100],
            "max_features": ["auto", None, 0.8],
            "max_depth": [None, 10, 100],
            "min_samples_leaf": [1, 2, 10],
        }
        rand_forest = RandomForestClassifier()
        g_search = GridSearchCV(rand_forest, param_grid,
                                cv=5, scoring=SCORING)
        g_search.fit(feature, target)
        logger.info("Random Forest "+str(g_search.best_params_))
        return g_search.best_estimator_


class GradientBoostingExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = {
            "n_estimators": [100, 1000],
            "max_features": ["auto", None],
            "max_depth": [3, 5, 10],
            "min_samples_leaf": [1, 2, 10],
        }
        g_boosting = GradientBoostingClassifier()
        g_search = GridSearchCV(g_boosting, param_grid, cv=5, scoring=SCORING)
        g_search.fit(feature, target),
        logger.info("GradientBoosting "+str(g_search.best_params_))
        return g_search.best_estimator_


class CombinedExecutor(BaseTrainExecutor):
    def fit(X):
        pass
