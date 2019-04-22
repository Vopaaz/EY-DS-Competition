# fill_0 best: {'max_depth': 10, 'max_features': 0.8, 'min_samples_leaf': 2, 'n_estimators': 100}
random_forest_1 = {
    "n_estimators": [10, 100],
    "max_features": ["auto", None, 0.8],
    "max_depth": [None, 10, 100],
    "min_samples_leaf": [1, 2, 10],
}

# fill_0 best: {'max_depth': 10, 'max_features': 0.8, 'min_samples_leaf': 2, 'n_estimators': 100}
random_forest_2 = {
    "n_estimators": [50, 100, 500],
    "max_features": [0.8, 0.9],
    "max_depth": [5, 10, 20],
    "min_samples_leaf": [2, 5]
}

# fill_0 best: {'max_depth': 5, 'max_features': None, 'min_samples_leaf': 10, 'n_estimators': 100}
gradient_boosting_1 = {
    "n_estimators": [100, 1000],
    "max_features": ["auto", None],
    "max_depth": [3, 5, 10],
    "min_samples_leaf": [1, 2, 10],
}

# fill_0 best: {'max_depth': 5, 'max_features': None, 'min_samples_leaf': 10, 'n_estimators': 100}
gradient_boosting_2 = {
    "n_estimators": [50, 100, 300],
    "max_features": [None],
    "max_depth": [5],
    "min_samples_leaf": [5, 10, 20]
}

SVC_1 = {
    "C": [1.0, 0.75, 1.25, 0.5, 1.5],
    "kernel": ["rbf", "sigmoid", "poly"],
    "gamma": ["auto", "scale"]
}
