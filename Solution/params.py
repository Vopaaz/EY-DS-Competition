# Final params: {'max_depth': 10, 'max_features': 0.8, 'min_samples_leaf': 2, 'n_estimators': 100}
random_forest_1 = {
    "n_estimators": [10, 100, 100],
    "max_features": ["auto", None, 0.8],
    "max_depth": [None, 10, 100],
    "min_samples_leaf": [1, 2, 10],
}

# Final params: {'max_depth': 5, 'max_features': None, 'min_samples_leaf': 10, 'n_estimators': 100}
gradient_boosting_1 = {
    "n_estimators": [100, 1000],
    "max_features": ["auto", None],
    "max_depth": [3, 5, 10],
    "min_samples_leaf": [1, 2, 10],
}
