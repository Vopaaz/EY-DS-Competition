import sys
sys.path.append(".")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from Solution.deeputil.Matrixfy import MatrixfyTransformer
from xgboost import XGBClassifier
from Solution.util.Submission import Submitter
from Solution.util.BaseUtil import Raw_DF_Reader
from Solution.util.Labelling import Labeller
from Solution.util.PathFilling import FillPathTransformer

import matplotlib.pyplot as plt

def provide_array():
    reader = Raw_DF_Reader()
    train = reader.train
    test = reader.test

    label = Labeller().transform(train).values
    label = label.reshape(label.shape[0],)

    filler = FillPathTransformer()
    train = filler.transform(train)
    test = filler.transform(test)

    print("Path filled.")

    matrixfier = MatrixfyTransformer(pixel=1000)
    matrixfier.fit(train, test)


    train_maps = matrixfier.transform(train)
    train_maps = np.array(list(train_maps.map_))

    test_maps = matrixfier.transform(test)
    test_maps = np.array(list(test_maps.map_))

    print("Matrixfied, resolution:", matrixfier.resolution)

    train_maps = train_maps.reshape(
        train_maps.shape[0], matrixfier.resolution[0] * matrixfier.resolution[1])
    test_maps = test_maps.reshape(
        test_maps.shape[0], matrixfier.resolution[0] * matrixfier.resolution[1])

    print("Reshape finished.")

    return train_maps, test_maps, label


def save(result):

    reader = Raw_DF_Reader()
    test = reader.test

    result = pd.DataFrame(result, columns=["target"])
    result["hash"] = test.hash.drop_duplicates().reset_index(drop=True)

    s = Submitter(result)
    s.save("Matrixfy PCA Approach, using full train set and PATH FILLED, pixel=1000.")


if __name__ == "__main__":
    train, test, label = provide_array()
    print("Array got.")
    pca = PCA(n_components=50, svd_solver="randomized")
    train = pca.fit_transform(train)
    test = pca.transform(test)
    print("PCA finish.")
    model = XGBClassifier()
    model.fit(train, label)
    print("XGB fit done.")
    res = model.predict(test)
    save(res)
