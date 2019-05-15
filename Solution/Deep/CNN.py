import sys
sys.path.append(".")
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras.utils import to_categorical
from Solution.deeputil.Matrixfy import MatrixfyTransformer
from Solution.util.Labelling import Labeller
from Solution.util.PathFilling import FillPathTransformer
from Solution.util.BaseUtil import Raw_DF_Reader, time_delta
from Solution.util.Submission import Submitter
from Solution.deeputil.MatrixProvider import MProvider
from Solution.deeputil.ValueFunc import naive_value


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

    def __init__(self, fill_path=True, pixel=1000, value_func=naive_value):
        train_provider = MProvider(
            "train", pixel=pixel, fill_path=fill_path, value_func=value_func)
        test_provider = MProvider(
            "test", pixel=pixel, fill_path=fill_path, value_func=value_func)

        r = Raw_DF_Reader()
        self._test = r.test
        self._train = r.train

        train_maps = train_provider.provide_matrix_df()
        test_maps = test_provider.provide_matrix_df()

        print("Initial matrix provided.")

        self.resolution = train_provider.resolution

        train_maps = np.array(list(train_maps.map_))
        test_maps = np.array(list(test_maps.map_))

        self.train_maps = train_maps.reshape(
            train_maps.shape[0], *train_provider.resolution, 1)
        self.test_maps = test_maps.reshape(
            test_maps.shape[0], *test_provider.resolution, 1)

        print("Matrix converted to 4D tensor.")

        labels = Labeller().transform(self._train)
        self.labels = to_categorical(list(labels.target))
        print("Label preparation completed.")

    def transform_result(self, result):
        '''
            Transfer the prediction result of CNN Dense layer to a pre-submittable DataFrame indexed by hash.
            Parameters:
                - result: a numpy array shaped (*, 2) derived from the CNN model

            Returns:
                - The prediction result DataFrame with two columns, "hash" and "target".
        '''

        result = pd.DataFrame(result, index=self._test.hash.drop_duplicates())
        result['target'] = result.apply(
            lambda series: 0 if series.iloc[0] >= 0.5 else 1, axis=1)
        result = result[["target"]]

        return result.reset_index().rename({"index": "hash"})


def init_model(resolution):
    '''
        Initialize the CNN model.
    '''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            input_shape=(*resolution, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
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
    '''
        Perform the main task.
    '''
    coor = CNNCoordinator()
    train_maps = coor.train_maps
    test_maps = coor.test_maps
    labels = coor.labels
    resolution = coor.resolution

    model = init_model(resolution)
    model.fit(train_maps, labels, epochs=30, batch_size=64)

    result = model.predict(test_maps)

    print(model.summary())

    res = coor.transform_result(result)
    s = Submitter(res)
    s.save("CNN 1st exploration")


if __name__ == "__main__":
    main()
