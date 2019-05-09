import logging

from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from Solution.Machine.Coordination import BaseTrainExecutor
from Solution.Machine.initLogging import init_logging
from Solution.Machine.params import (SVC_1, SVC_fill_0_best, XGBoosting_1,
                                     gradient_boosting_2,
                                     gradient_boosting_fill_0_best,
                                     random_forest_2,
                                     random_forest_fill_0_best,
                                     XGBoosting_2, random_forest_3)
from xgboost import XGBClassifier

logger = init_logging()
SCORING = make_scorer(f1_score)


class RandomForestExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = random_forest_3
        rand_forest = RandomForestClassifier()
        g_search = GridSearchCV(rand_forest, param_grid,
                                cv=5, scoring=SCORING)
        g_search.fit(feature, target)
        logger.info("Random Forest, drop, 3rd params. Final trial" + str(g_search.best_params_))
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


class SupportVectorExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = SVC_1
        svc = SVC()
        g_search = GridSearchCV(svc, param_grid, cv=5, scoring=SCORING)
        g_search.fit(feature, target)
        logger.info("SVC "+str(g_search.best_params_))
        return g_search.best_estimator_


class XGBoostExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = XGBoosting_2
        xgb = XGBClassifier()
        g_search = GridSearchCV(xgb, param_grid, cv=5, scoring=SCORING)
        g_search.fit(feature, target)
        logger.info("XGBoosting, separate_all strat, column num {}, param_grid using [2]".format(X.shape[1])+str(g_search.best_params_))
        return g_search.best_estimator_


class CombinedExecutor(BaseTrainExecutor):
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        rf = RandomForestClassifier(**random_forest_fill_0_best)
        gb = GradientBoostingClassifier(**gradient_boosting_fill_0_best)
        vot = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        return vot.fit(feature, target)
