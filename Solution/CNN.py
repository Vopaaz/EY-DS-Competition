from util.utilFunc import Raw_DF_Reader, time_delta

from deeputil.Matrixfy import MatrixfyTransformer
import pandas as pd
from util.Labelling import Labeller

from keras import layers
from keras import models
from keras.utils import to_categorical
import numpy as np

from util.PathFilling import FillPathTransformer


def naive_value(timestamp):
    start = pd.Timestamp("1900-01-01 00:00:00")
    end = pd.Timestamp("1900-01-01 23:59:59")
    return time_delta(timestamp, start) / time_delta(start, end)


def main():
    import matplotlib.pyplot as plt

    r = Raw_DF_Reader()
    train = r.train.iloc[:41]
    test = r.test.iloc[:34]

    # train = FillPathTransformer().transform(train)
    # test = FillPathTransformer().transform(test)

    t = MatrixfyTransformer(100, naive_value)
    t.fit(train, test)

    train_maps = t.transform(train)
    test_maps = t.transform(test)

    train_maps = np.array(list(train_maps.map_))
    test_maps = np.array(list(test_maps.map_))

    # plt.imshow(train_maps[0], cmap=plt.cm.binary)
    # plt.show()
    # plt.imshow(test_maps[0], cmap=plt.cm.binary)
    # plt.show()

    train_maps = train_maps.reshape(train_maps.shape[0], *t.resolution, 1)
    test_maps = test_maps.reshape(test_maps.shape[0], *t.resolution, 1)

    label = Labeller().transform(train)

    print(label)

    label = to_categorical(list(label.target))

    print(train_maps.shape)
    print(t.resolution)
    print(label)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            input_shape=(*t.resolution, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_maps, label, epochs=1, batch_size=64)

    res = model.predict(test_maps)
    print(res)

    res = pd.DataFrame(res, index=test.hash.drop_duplicates())
    res['target'] = res.apply(
        lambda series: 0 if series.iloc[0] >= 0.5 else 1, axis=1)
    res = res[["target"]]
    print(res)

if __name__ == "__main__":
    main()
