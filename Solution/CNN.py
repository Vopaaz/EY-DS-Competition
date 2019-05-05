import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras.utils import to_categorical

from deeputil.Matrixfy import MatrixfyTransformer
from util.Labelling import Labeller
from util.PathFilling import FillPathTransformer
from util.utilFunc import Raw_DF_Reader, time_delta


def naive_value(timestamp):
    start = pd.Timestamp("1900-01-01 00:00:00")
    end = pd.Timestamp("1900-01-01 23:59:59")
    return time_delta(timestamp, start) / time_delta(start, end)


class CNNCoordinator(object):
    '''
        Prepare the input tensor and label for CNN.
        Parameters:
            - fill_path: boolean, whether or not to let the map be the path-filled version.
            - pixel: the pixel parameter passed to the MatrixfyTransformer, representing the width and height for one pixel in the map.
            - value_func: the value function passed to the MatrixfyTransformer.

        Attributes:
            - train_maps: A 4D tensor, (samples, height, width, channel),
                the channel has only one dimension (grayscale images) for the train set.
            - test_maps: The corresponding tensor for the test set.
            - label: The one-hot encoded label for the train set.
            - resolution: The resolution of the label (The number of pixels in height and width)
    '''

    def __init__(self, fill_path=True, pixel=100, value_func=naive_value):
        r = Raw_DF_Reader()
        train = r.train.iloc[:102]
        test = r.test.iloc[:18]
        self._test = test

        if fill_path:
            train = FillPathTransformer().transform(train)
            test = FillPathTransformer().transform(test)
            print("Path filled.")

        t = MatrixfyTransformer(pixel, value_func)
        t.fit(train, test)
        train_maps = t.transform(train)
        test_maps = t.transform(test)

        print("Initial matrix provided.")

        self.resolution = t.resolution

        train_maps = np.array(list(train_maps.map_))
        test_maps = np.array(list(test_maps.map_))

        self.train_maps = train_maps.reshape(
            train_maps.shape[0], *t.resolution, 1)
        self.test_maps = test_maps.reshape(
            test_maps.shape[0], *t.resolution, 1)

        print("Matrix converted to 4D tensor.")

        labels = Labeller().transform(train)
        self.labels = to_categorical(list(labels.target))
        print("Label preparation completed.")

    def transform_result(self, result):
        '''
            Transfer the prediction result of CNN Dense layer to a pre-submittable DataFrame indexed by hash.
            Parameters:
                - result: a numpy array shaped (*, 2) derived from the CNN model

            Returns:
                - a hash-indexed DataFrame with one column as the prediction result.
        '''

        result = pd.DataFrame(result, index=self._test.hash.drop_duplicates())
        result['target'] = result.apply(
            lambda series: 0 if series.iloc[0] >= 0.5 else 1, axis=1)
        result = result[["target"]]

        return result


def init_model(resolution):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            input_shape=(*resolution, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(2, activation="sigmoid"))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def main():
    coor = CNNCoordinator()
    train_maps = coor.train_maps
    test_maps = coor.test_maps
    labels = coor.labels
    resolution = coor.resolution

    model = init_model(resolution)

    model.fit(train_maps, labels, epochs=3, batch_size=64)

    result = model.predict(test_maps)

    print(coor.transform_result(result))


if __name__ == "__main__":
    main()
