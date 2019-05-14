'''
    Hyper-parameters used for grid search.
'''

# fill_0 best: {'max_depth': 10, 'max_features': 0.8, 'min_samples_leaf': 2, 'n_estimators': 100}
random_forest_1 = {
    "n_estimators": [10, 100],
    "max_features": ["auto", None, 0.8],
    "max_depth": [None, 10, 100],
    "min_samples_leaf": [1, 2, 10],
}

# fill_0 best: {'max_depth': 10, 'max_features': 0.8, 'min_samples_leaf': 2, 'n_estimators': 100}
# drop best: {'max_depth': 10, 'max_features': 0.8, 'min_samples_leaf': 2, 'n_estimators': 100}
# drop with outliers killed best:{'max_depth': 10, 'max_features': 0.9, 'min_samples_leaf': 5, 'n_estimators': 100}
random_forest_2 = {
    "n_estimators": [50, 100, 500],
    "max_features": [0.8, 0.9],
    "max_depth": [5, 10, 20],
    "min_samples_leaf": [2, 5]
}

random_forest_3 = {
    "n_estimators": [75, 100, 150],
    "max_features": [0.85, 0.9, 0.95],
    "max_depth": [8, 10, 15],
    "min_samples_leaf": [4, 5, 6]
}

# fill_0 best: {'max_depth': 5, 'max_features': None, 'min_samples_leaf': 10, 'n_estimators': 100}
gradient_boosting_1 = {
    "n_estimators": [100, 1000],
    "max_features": ["auto", None],
    "max_depth": [3, 5, 10],
    "min_samples_leaf": [1, 2, 10],
}

# fill_0 best: {'max_depth': 5, 'max_features': None, 'min_samples_leaf': 10, 'n_estimators': 100}
# drop best: {'max_depth': 5, 'max_features': None, 'min_samples_leaf': 20, 'n_estimators': 100}
gradient_boosting_2 = {
    "n_estimators": [50, 100, 300],
    "max_features": [None],
    "max_depth": [5],
    "min_samples_leaf": [5, 10, 20]
}

# fill_0 best: {'C': 1.5, 'gamma': 'auto', 'kernel': 'rbf'}
SVC_1 = {
    "C": [1.0, 0.75, 1.25, 0.5, 1.5],
    "kernel": ["rbf", "sigmoid", "poly"],
    "gamma": ["auto", "scale"]
}


random_forest_fill_0_best = {'max_depth': 10, 'max_features': 0.8,
                             'min_samples_leaf': 2, 'n_estimators': 100}

gradient_boosting_fill_0_best = {
    'max_depth': 5, 'max_features': None, 'min_samples_leaf': 10, 'n_estimators': 100}

SVC_fill_0_best = {'C': 1.5, 'gamma': 'auto',
                   'kernel': 'rbf', 'probability': True}

# drop best: {'gamma': 0, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200}
# drop with outlier-elimination: {'gamma': 0, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 50}
XGBoosting_1 = {
    "max_depth": [2, 3, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "gamma": [0, 0.01, 0.05],
}

XGBoosting_2 = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [25, 50, 100],
    "gamma": [0, 0.01]
}
