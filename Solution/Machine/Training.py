'''
    The machine learning models trainner.
'''

import logging

from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from Solution.Machine.Coordination import BaseTrainExecutor
from Solution.Machine.params import (SVC_1, SVC_fill_0_best, XGBoosting_1,
                                     gradient_boosting_2,
                                     gradient_boosting_fill_0_best,
                                     random_forest_2,
                                     random_forest_fill_0_best,
                                     XGBoosting_2, random_forest_3)
from xgboost import XGBClassifier


class RandomForestExecutor(BaseTrainExecutor):
    '''
        Wrap the RandomForestClassifer.
    '''
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = random_forest_3
        rand_forest = RandomForestClassifier()
        g_search = GridSearchCV(rand_forest, param_grid,
                                cv=5, scoring=self.SCORING)
        g_search.fit(feature, target)
        self.logger.info("Random Forest, drop, 3rd params. Final trial" +
                         str(g_search.best_params_))
        return g_search.best_estimator_


class GradientBoostingExecutor(BaseTrainExecutor):
    '''
        Wrap the GradientBoostingClassifier.
    '''
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = gradient_boosting_2
        g_boosting = GradientBoostingClassifier()
        g_search = GridSearchCV(g_boosting, param_grid,
                                cv=5, scoring=self.SCORING)
        g_search.fit(feature, target),
        self.logger.info("GradientBoosting "+str(g_search.best_params_))
        return g_search.best_estimator_


class SupportVectorExecutor(BaseTrainExecutor):
    '''
        Wrap the SupportVectorClassifier.
    '''
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = SVC_1
        svc = SVC()
        g_search = GridSearchCV(svc, param_grid, cv=5, scoring=self.SCORING)
        g_search.fit(feature, target)
        self.logger.info("SVC "+str(g_search.best_params_))
        return g_search.best_estimator_


class XGBoostExecutor(BaseTrainExecutor):
    '''
        Wrap the XGBoostingClassifier
    '''
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        param_grid = XGBoosting_2
        xgb = XGBClassifier()
        g_search = GridSearchCV(xgb, param_grid, cv=5, scoring=self.SCORING)
        g_search.fit(feature, target)
        self.logger.info("XGBoosting, Drop strategy. Best Parameters: " +
                         str(g_search.best_params_))
        return g_search.best_estimator_


class CombinedExecutor(BaseTrainExecutor):
    '''
        Wrap the VotingClassifier using RandomForestClassifier and GradientBoostingClassifier.
    '''
    def fit(self, X):
        _, feature, target = self.split_hash_feature_target(X)
        rf = RandomForestClassifier(**random_forest_fill_0_best)
        gb = GradientBoostingClassifier(**gradient_boosting_fill_0_best)
        vot = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        return vot.fit(feature, target)
