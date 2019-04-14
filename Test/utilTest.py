import sys
sys.path.append(".")
sys.path.append(r".\Solution")

import unittest
import pandas as pd
import numpy as np
from Solution.NaiveFeature import NaiveDistanceExtractor
from Solution.util import distance_to_border, isin_center


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


class NaiveDistanceTest(unittest.TestCase):
    def test_filled_case(self):
        t = NaiveDistanceExtractor(path_filled=True)
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

    def test_not_filled_case(self):
        t = NaiveDistanceExtractor(path_filled=False)
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


if __name__ == "__main__":
    unittest.main()
