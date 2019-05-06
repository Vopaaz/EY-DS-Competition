import sys
sys.path.append(".")
sys.path.append(r".\Solution")
from Solution.util.Labelling import Labeller
from Solution.util.BaseUtil import distance_to_border, isin_center, distance_between, time_delta
from Solution.util.NaiveFeature import DistanceInfoExtractor, PathInfoExtractor, CoordinateInfoExtractor
import numpy as np
import pandas as pd
import unittest

MIN_X = 3750901.5068
MAX_X = 3770901.5068
MIN_Y = -19268905.6133
MAX_Y = -19208905.6133


class UtilTest(unittest.TestCase):
    def test_isin_center(self):
        self.assertEqual(isin_center(0, 0), 0)
        self.assertEqual(isin_center(3750901.5069, -19268905.6132), 1)
        x = pd.Series([0, 3750901.5069, 123])
        y = pd.Series([0, -19268905.6132, -123])
        self.assertSequenceEqual(list(isin_center(x, y)), [0, 1, 0])

    def test_distance_to_border(self):
        # Case of four out-border corner
        self.assertEqual(distance_to_border(MIN_X-1, MIN_Y-1), 2)
        self.assertEqual(distance_to_border(MIN_X-1, MAX_Y+2), 3)
        self.assertEqual(distance_to_border(MAX_X+2, MAX_Y+3), 5)
        self.assertEqual(distance_to_border(MAX_X+3, MIN_Y-3), 6)

        # Case that in the border
        self.assertEqual(distance_to_border(MIN_X+1, MIN_Y+2), -1)
        self.assertEqual(distance_to_border(MAX_X-10, MAX_Y-1), -1)
        self.assertEqual(distance_to_border(MAX_X-3, MAX_Y-15), -3)

        # Case for the "Cross"
        self.assertEqual(distance_to_border(MIN_X+5, MAX_Y+10), 10)
        self.assertEqual(distance_to_border(MIN_X+5, MIN_Y-20), 20)
        self.assertEqual(distance_to_border(MIN_X-7, MIN_Y+20), 7)
        self.assertEqual(distance_to_border(MAX_X+25, MIN_Y+20), 25)

        # Case for pd.Series
        x = pd.Series([MIN_X-1, MIN_X+1, MIN_X+5])
        y = pd.Series([MIN_Y-1, MIN_Y+2, MAX_Y+10])
        self.assertSequenceEqual(list(distance_to_border(x, y)), [2, -1, 10])

    def test_distance_between_points(self):
        self.assertEqual(distance_between(0, 0, 2, 2), 4)
        self.assertEqual(distance_between(0, 0, 2, 2), 4)
        self.assertEqual(distance_between(0, 10, 0, 5), 5)
        x1 = pd.Series([0, 1])
        y1 = pd.Series([0, -1])
        x2 = pd.Series([3, 4])
        y2 = pd.Series([5, 4])
        self.assertSequenceEqual(
            list(distance_between(x1, y1, x2, y2)), [8, 8])

    def test_time_delta(self):
        ts1 = pd.Timestamp("2019-4-16 12:00:00")
        ts2 = pd.Timestamp("2019-4-16 12:01:00")

        ts_series1 = pd.Series([ts1, ts2])
        ts_series2 = pd.Series([ts2, ts1])

        self.assertEqual(time_delta(ts1, ts2), 60)
        self.assertSequenceEqual(
            list(time_delta(ts_series1, ts_series2)), [60, 60])


class DistanceInfoTest(unittest.TestCase):
    def test_filled_case(self):
        t = DistanceInfoExtractor(path_filled=True)
        test_df = pd.DataFrame({
            "hash": ["1", "1", "2", "2"],
            "x_entry": [MIN_X, MIN_X+1, MAX_X-1, MAX_X+1],
            "y_entry": [MIN_Y, MIN_Y+1, MIN_Y, MIN_Y],
            "x_exit": [MIN_X+1, np.nan, MAX_X+1, np.nan],
            "y_exit": [MIN_Y+1, np.nan, MIN_Y, np.nan]
        })
        res = t.fit_transform(test_df)
        self.assertSequenceEqual(list(res.max_distance), [0, 1])
        self.assertSequenceEqual(list(res.min_distance), [-1, 0])
        self.assertSequenceEqual(list(res.avg_distance), [-0.5, 0.5])
        self.assertSequenceEqual(list(res.start_end_dist_diff), [-1, 1])
        self.assertSequenceEqual(list(res.last_path_dist_diff), [-1, 1])

    def test_not_filled_case(self):
        t = DistanceInfoExtractor(path_filled=False)
        test_df = pd.DataFrame({
            "hash": ["1", "1", "2", "2"],
            "x_entry": [MIN_X, MIN_X+1, MAX_X-1, MAX_X+1],
            "y_entry": [MIN_Y, MIN_Y+1, MIN_Y, MIN_Y],
            "x_exit": [MIN_X+2, np.nan, MAX_X+2, np.nan],
            "y_exit": [MIN_Y+2, np.nan, MIN_Y, np.nan]
        })
        res = t.fit_transform(test_df)
        self.assertSequenceEqual(list(res.max_distance), [0, 2])
        self.assertSequenceEqual(list(res.min_distance), [-2, 0])
        self.assertSequenceEqual(list(res.avg_distance), [-1, 1])
        self.assertSequenceEqual(list(res.start_end_dist_diff), [-2, 2])
        self.assertSequenceEqual(list(res.last_path_dist_diff), [-2, 2])


class PathInfoTest(unittest.TestCase):
    def test_path_info(self):
        test_df = pd.DataFrame({
            "hash": ["1", "1", "1", "2", "2", "2"],
            "x_entry": [0, 0, 0, 0, 0, 0],
            "y_entry": [0, 0, 0, 0, 0, 0],
            "x_exit": [0, 0, 0, 0, 0, 0],
            "y_exit": [1, 2, 3, 4, 5, 6],
            "time_entry": [
                pd.Timestamp("2000-01-01 00:00:00"),
                pd.Timestamp("2000-01-01 00:01:00"),
                pd.Timestamp("2000-01-01 00:01:00"),
                pd.Timestamp("2000-01-01 00:00:00"),
                pd.Timestamp("2000-01-01 00:01:00"),
                pd.Timestamp("2000-01-01 00:02:00"),
            ],
            "time_exit": [
                pd.Timestamp("2000-01-01 00:00:00"),
                pd.Timestamp("2000-01-01 00:01:01"),
                pd.Timestamp("2000-01-01 00:02:00"),
                pd.Timestamp("2000-01-01 00:00:01"),
                pd.Timestamp("2000-01-01 00:01:05"),
                pd.Timestamp("2000-01-01 00:02:10"),
            ]
        })
        res = PathInfoExtractor().fit_transform(test_df)
        self.assertSequenceEqual(list(res.max_length), [2, 5])
        self.assertSequenceEqual(list(res.min_length), [1, 4])
        self.assertSequenceEqual(list(res.avg_length), [1.5, 4.5])
        self.assertSequenceEqual(list(res.max_velocity), [2, 4])
        self.assertSequenceEqual(list(res.min_velocity), [2, 1])
        self.assertSequenceEqual(list(res.avg_velocity), [2, 2.5])


class CoordinateInfoTest(unittest.TestCase):
    def test_last_coordinate(self):
        test_df = pd.DataFrame({
            "hash": ["1", "1", "1", "2", "2", "2"],
            "x_entry": [0, 1, 2, 3, 4, 5],
            "y_entry": [6, 5, 4, 3, 2, 1]
        })
        res = CoordinateInfoExtractor().fit_transform(test_df)
        self.assertSequenceEqual(list(res.x_last_point), [2, 5])
        self.assertSequenceEqual(list(res.y_last_point), [4, 1])


class LabelTest(unittest.TestCase):
    def test_labelling(self):
        test_df = pd.DataFrame({
            "hash": ["1", "1", "2", "2"],
            "x_exit": [np.nan, MIN_X+1, np.nan, MAX_X+1],
            "y_exit": [np.nan, MIN_Y, np.nan, MIN_Y]
        })
        res = Labeller().fit_transform(test_df)
        self.assertSequenceEqual(list(res.target), [1, 0])


if __name__ == "__main__":
    unittest.main()
