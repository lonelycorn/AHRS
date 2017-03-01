import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np

from estimation.kalman_filter import KalmanFilterSO3
from base.utility import test_matrix_equal


class TestKalmanFilterSO3(unittest.TestCase):

    VALUE_EQUAL_PLACE = 2

    def setUp(self):
        # define some initial states
        self._R0 = np.eye(3)
        self._P0 = np.zeros((3, 3))
        self._gyro_cov = 0.01 * np.eye(3)
        self._mag_cov = 0.01 * np.eye(3)
        self._acc_cov = 0.01 * np.eye(3)
        self._gravity = np.array([0, 0, -1.0])
        self._mag_field = np.array([1.0, 0.0, 0.0])
        self._dt = 0.1

        self._kf = KalmanFilterSO3(self._R0, self._P0, self._gyro_cov, self._acc_cov,
                                   self._mag_cov, self._gravity, self._mag_field)

    def test_process_update(self):
        # define an angular velocity
        om_meas = np.array([0.0, 0.0, 1.0])
        self._kf.process_update(om_meas, self._dt)

        R_est = self._kf.get_estimate_mean().get_matrix()
        P_est = self._kf.get_estimate_covar()

        R_true = np.array([[np.cos(0.1), np.sin(0.1), 0.0],
                           [-np.sin(0.1), np.cos(0.1), 0.0],
                           [0.0, 0.0, 1.0]])
        P_true = 0.01 * 0.01 * np.eye(3)

        result = test_matrix_equal(R_est, R_true)
        self.assertTrue(result, "rotations do not match for process update!")
        result = test_matrix_equal(P_est, P_true)
        self.assertTrue(result, "covariances do not match for process update!")

    def test_acc_update(self):
        # TODO: just testing if measurement update can run and output something
        # TODO: no value checking for now
        acc_meas = np.array([0.0, -1.0, 0.0])
        self._kf.acc_update(acc_meas)

        mag_meas = np.array([0.0, 1.0, 0.0])
        self._kf.mag_update(mag_meas)

        self.assertTrue(True)


if (__name__ == "__main__"):
    unittest.main()
