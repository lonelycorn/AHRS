import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np

from base.least_square_estimator import LeastSquareEstimator

class TestLeastSquareEstimator(unittest.TestCase):
    '''
    NOTE: not testing the forgetting factor now.
    '''
    VALUE_EQUAL_PLACE = 2
    def setUp(self):
        pass

    def test_straight_line_no_noise(self):
        '''
        2D points are generated according to y = k * x + b, without noise.
        '''
        k = 1.0
        b = 10.0
        x_sample = np.arange(0, 100, 1)
        y_sample = k * x_sample + b

        initial_value = np.array([0.0, 0.0]) # deliberately made off
        initial_covar = np.eye(2) * 1e3
        lse = LeastSquareEstimator(initial_value, initial_covar)

        for (x, y) in zip(x_sample, y_sample):
            phi = np.array([x, 1])
            lse.update(phi, y)

        mean = lse.get_estimate_mean()

        self.assertAlmostEqual(mean[0], k, TestLeastSquareEstimator.VALUE_EQUAL_PLACE)
        self.assertAlmostEqual(mean[1], b, TestLeastSquareEstimator.VALUE_EQUAL_PLACE)

    def test_straight_line_symmetric(self):
        '''
        2D points are symmetric about y = 0.
        '''
        x_sample = np.arange(-50, 50, 1)
        y_sample = 2 * (np.mod(x_sample, 2) - 0.5)

        initial_value = np.array([1.0, -1.0]) # deliberately made off
        initial_covar = np.eye(2) * 1e3
        lse = LeastSquareEstimator(initial_value, initial_covar)

        for (x, y) in zip(x_sample, y_sample):
            phi = np.array([x, 1])
            lse.update(phi, y)

        mean = lse.get_estimate_mean()

        self.assertAlmostEqual(mean[0], 0.0, TestLeastSquareEstimator.VALUE_EQUAL_PLACE)
        self.assertAlmostEqual(mean[1], 0.0, TestLeastSquareEstimator.VALUE_EQUAL_PLACE)

    def test_strait_line(self):
        '''
        2D points are generated according to y = k * x + b, with moderate noise
        '''
        k = 1.0
        b = 10.0
        x_sample = np.arange(0, 100, 1)
        y_sample = k * x_sample + b + np.random.normal(0.0, 0.01, x_sample.shape)

        initial_value = np.array([0.0, 0.0]) # deliberately made off
        initial_covar = np.eye(2) * 1e3
        lse = LeastSquareEstimator(initial_value, initial_covar)

        for (x, y) in zip(x_sample, y_sample):
            phi = np.array([x, 1])
            lse.update(phi, y)

        mean = lse.get_estimate_mean()

        self.assertAlmostEqual(mean[0], k, TestLeastSquareEstimator.VALUE_EQUAL_PLACE)
        self.assertAlmostEqual(mean[1], b, TestLeastSquareEstimator.VALUE_EQUAL_PLACE)

if (__name__ == "__main__"):
    unittest.main()
