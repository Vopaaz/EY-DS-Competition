# EY Nextwave Data Science Challenge 2019 Solution

Solution for 2019 EY Nextwave Data Science Challenge by Vopaaz and Xiaochr.


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [EY Nextwave Data Science Challenge 2019 Solution](#ey-nextwave-data-science-challenge-2019-solution)
  - [Prerequisites](#prerequisites)
    - [Environment](#environment)
    - [Installing Dependencies](#installing-dependencies)
    - [Prepare Data](#prepare-data)
  - [Getting the Final Result](#getting-the-final-result)
  - [Methodology](#methodology)
    - [Used in the Final Result](#used-in-the-final-result)
      - [Feature Engineering](#feature-engineering)
    - [Explored but not Used](#explored-but-not-used)
      - [Feature Engineering](#feature-engineering-1)

<!-- /code_chunk_output -->


## Prerequisites

### Environment

`Python 3.7.2 64-bit`

### Installing Dependencies

Use virtrual environment if necessary.

```bash
$ pip install -r requirements.txt
```

### Prepare Data

Place the train dataset in `OriginalFile/data_train/data_train.csv`.

Place the test dataset in `OriginalFile/data_test/data_test.csv`.

## Getting the Final Result

```bash
$ python Solution/FinalResult.py
```

This shall take about 1-2 hours. Then the `.csv` file to be submitted can be found in directory `Result/`.

Note that as we did neither save the model nor set the `random_state` variable, the produced file may be slightly different from our last submission.

## Methodology

### Used in the Final Result

#### Feature Engineering

The paths given in the datasets are not fully connected. However, logically they should be connected end to end.
Thus we firstly joined all the disconnected paths and do the following feature extraction.

We listed features of each device that we think may affect the prediction target (i.e. whether the last exit point of this device is within the city center or not), they are:

- The difference between 3 p.m. and the starting / ending time point of the last path. (in seconds)
- The difference between the starting and ending time point of the last path. (in seconds)
- The max, min, average level of the distance of all the points recorded by a device.
- The difference between the distance of the entry of the first path and the exit of the last but one.
- The difference between the distance of the entry and the exit of the last path but one.
- The min, max, average level of the length of all the paths recorded by a device
- The min, max, average level of the average velocity of all the paths recorded by a device
- The coordinate of the start point of the last path (the path to be predicted).

There are some devices which only records one path (the path to be predicted). Hence some of the above-mentioned features cannot be extracted. They are `Null` values in the Feature Panel. We removed these features.

After having these features, we use the Isolation Forest to detect and remove the outliers in the train set, and then standardized the rest of them to have a standard normal distribution. The test data is also standardized.

### Explored but not Used

#### Feature Engineering

The traditional way of dealing with `Null` values are filling them with zeros or dropping them.
Nevertheless, the `Null` values in our Feature Panel are not caused by common problems such as record error, but rather because the number of path recorded is only one.

Based on the fact that there are considerable numbers of devices which has only one path recorded. We develop another two strategies to deal with the `Null` values.


