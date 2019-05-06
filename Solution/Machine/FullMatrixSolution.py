import sys
sys.path.append(".")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from Solution.deeputil.Matrixfy import MatrixfyTransformer
from xgboost import XGBClassifier
from Solution.util.Submition import Submitter
from Solution.util.BaseUtil import Raw_DF_Reader
from Solution.util.Labelling import Labeller

import matplotlib.pyplot as plt

def provide_array():
    reader = Raw_DF_Reader()
    train = reader.train
    test = reader.test

    matrixfier = MatrixfyTransformer(pixel=1000)
    matrixfier.fit(train, test)

    print(matrixfier.resolution)

    label = Labeller().transform(train).values
    label = label.reshape(label.shape[0],)

    train_maps = matrixfier.transform(train)
    train_maps = np.array(list(train_maps.map_))

    test_maps = matrixfier.transform(test)
    test_maps = np.array(list(test_maps.map_))

    train_maps = train_maps.reshape(
        train_maps.shape[0], matrixfier.resolution[0] * matrixfier.resolution[1])
    test_maps = test_maps.reshape(
        test_maps.shape[0], matrixfier.resolution[0] * matrixfier.resolution[1])

    return train_maps, test_maps, label


def save(result):

    reader = Raw_DF_Reader()
    test = reader.test

    result = pd.DataFrame(result, columns=["target"])
    result["hash"] = test.hash.drop_duplicates().reset_index(drop=True)

    s = Submitter(result)
    s.save("Matrixfy PCA Approach, using full train set, pixel=1000.")


if __name__ == "__main__":
    train, test, label = provide_array()
    pca = PCA(n_components=50, svd_solver="randomized")
    train = pca.fit_transform(train)
    test = pca.transform(test)
    model = XGBClassifier()
    model.fit(train, label)
    res = model.predict(test)
    save(res)
