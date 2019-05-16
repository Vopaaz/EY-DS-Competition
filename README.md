# EY Nextwave Data Science Challenge 2019 Solution

Solution for 2019 EY Nextwave Data Science Challenge by Vopaaz and Xiaochr.


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [EY Nextwave Data Science Challenge 2019 Solution](#ey-nextwave-data-science-challenge-2019-solution)
	- [Getting the Final Result](#getting-the-final-result)
		- [Prerequisites](#prerequisites)
			- [Environment](#environment)
			- [Installing Dependencies](#installing-dependencies)
		- [Prepare Data](#prepare-data)
		- [Running](#running)
	- [Methodology](#methodology)
		- [Feature Engineering](#feature-engineering)
		- [Null Value Feature Handling](#null-value-feature-handling)
		- [Algorithm Design](#algorithm-design)
			- [Preprocessing](#preprocessing)
			- [Model Training and Selection](#model-training-and-selection)
	- [Module Documentation](#module-documentation)
	- [Approach Explored but not Used](#approach-explored-but-not-used)

<!-- /code_chunk_output -->



## Getting the Final Result

### Prerequisites

#### Environment

`Python 3.7.2 64-bit`

#### Installing Dependencies

Use virtrual environment if necessary.

```bash
$ pip install -r requirements.txt
```

### Prepare Data

Place the train dataset in `OriginalFile/data_train/data_train.csv`.

Place the test dataset in `OriginalFile/data_test/data_test.csv`.

### Running

```bash
$ python Solution/FinalResult.py
```

This shall take about 1-2 hours. Then the `.csv` file to be submitted can be found in directory `Result/`.

Note that as we neither saved the model nor set the `random_state` variable, the produced file may be slightly different from our last submission.


## Methodology

### Feature Engineering

The paths given in the datasets are not fully connected. However, logically they should be connected end to end.
Thus we firstly joined all the disconnected paths and do the following feature extraction.

We listed features of each device that we think may affect the prediction target (i.e. whether the last exit point of this device is within the city center or not), they are:

- The difference between 3 p.m. and the starting / ending time point of the unknown path. (in seconds)
- The difference between the starting and ending time point of the unknown path. (in seconds)
- The max, min, average level of the distance to the central area of all the points recorded by a device.
- The difference between the distance to the central area of the entry of the first path and the exit of the last known path.
- The difference between the distance to the central area of the entry and the exit of the last known path.
- The min, max, average level of the length of all the paths recorded by a device
- The min, max, average level of the average velocity of all the paths recorded by a device
- The coordinate of the start point of the unknown path

All the *distance to the central area* are measured by the l1 distance to the border of the central area.

There are some devices which only records one path (the path to be predicted). Hence some of the above-mentioned features cannot be extracted. They are `Null` values in the Feature Panel.
We came up with several strategies to deal with them (see the [Null Value Feature Handling](#null-value-feature-handling) part). In the best prediction result, we used the `drop` strategy, that is, to remove these features.


### Null Value Feature Handling

The traditional way of dealing with `Null` values are filling them with zeros or dropping them.
Nevertheless, the `Null` values in our Feature Panel are not caused by common problems such as record error, but rather because the number of path recorded is only one.

Based on the fact that there are considerable numbers of devices which are in this case. We develop another two strategies to deal with the `Null` values.

Example DataFrame (`v` means valid value and `N` means `Null`, each column is a feature and each row is a device):

```
    A   B   C
0   v   v   v
1   v   v   v
2   v   v   N
3   v   v   N
4   v   N   N
5   v   N   N
```

`separate_all` strategy:

- Use `(0-5).A` to train the model and predict those whose non-null feature is only `A`
- Use `(0-3).AB` to train the model and predict those whose non-null feature is `A, B`
- Use `(0-1).ABC` to train the model and predict those whose non-null feature is `A, B, C`

`separate_part` strategy:
- Use `(4-5).A` to train the model and predict those whose non-null feature is only `A`
- Use `(2-3).AB` to train the model and predict those whose non-null feature is `A, B`
- Use `(0-1).ABC` to train the model and predict those whose non-null feature is `A, B, C`

In this way we make full use of the extracted features. In some models, the `separate_all` strategy do provides a better result.


### Algorithm Design

After having the features, we applied general machine learning approaches to do the classification.

#### Preprocessing

We used the `Isolation Forest` to detect and removed the 5% outliers in the train set, and then standardized the rest of them to have a standard normal distribution. The test data was also standardized.

#### Model Training and Selection

We have tried the following models:

- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoosting

For each model, we ran several runs of grid search, each adjusted 3-4 hyper-parameters and gradually narrowed down the range. The cross validation splitting was set to `k=5`.

Finally, we selected the `XGBoosting` framework to provide the final submission based on the validation result.

The best hyper-parameters we found are:

- gamma: 0.01
- learning_rate: 0.1
- max_depth: 7
- n_estimators: 100
- others: default

Although we have found the best parameters, the script we provide still runs the final round of grid search because we try to reproduct the final submission. Also, the time required is relatively acceptable.


## Module Documentation

Please use the browser to open `Doc/Solution/index.html`.

## Approach Explored but not Used

The feature engineering process described in [Feature Engineering](#feature-engineering) were effective, but obviously some information are still lost. We believe that if we can preserve more information, the classification result will be better.

The intuitive method would be extract the location (coordinate) of each device at a series of time points. However, there are considerable numbers of devices who have only one path record. In this case, most coordinate values in the time series will be `Null`. Typical models cannot deal with it.

Another approach we have come up with is to convert the paths of one device into an image, by connecting the entry and exit point of one path on the map panel with straight line. The shade of each pixel on the line is determined by the esimated time when the device is located there. This is also intuitive.
We did not actually produce any image file but rather generated the equivalent matrix directly.

After having these matrices (equivalent images), we apply the convolutional neural network for the classical image classification problem. The final result was not satisfing, though. The best result we got from this method is around 0.801.

According to our analysis, the reason may be that CNN is designed to deal with real pictures, where the equivalent matrix is dense. Thus it can learn different local features from it. Our matrix, nevertheless, is quite sparse. It does not fit the application of CNN.

The code implementing this approach are in `Solution/Deep` and `Solution/deeputil`.

