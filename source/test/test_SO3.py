import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np

from base.SO3 import SO3
from base.utility import test_matrix_equal

default_tol_place = 2



class TestSO3(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_constructor(self):
        rot = SO3()
        result = test_matrix_equal(rot.get_matrix(), np.eye(3))
        self.assertTrue(result, "Default is not identity!")

    def test_euler_constructor(self):
        roll, pitch, yaw = (0.1, -0.2, 0.3)
        rot = SO3.from_euler(roll, pitch, yaw)

        self.assertAlmostEqual(rot.get_roll(), roll, places=default_tol_place)
        self.assertAlmostEqual(rot.get_pitch(), pitch, places=default_tol_place)
        self.assertAlmostEqual(rot.get_yaw(), yaw, places=default_tol_place)

    def test_two_directions_constructor_opposite(self):
        d_f = np.array([ -9.41427684e-03, -7.26582309e-03, 9.78452150e+00], dtype=np.float)
        d_t = np.array([0, 0, -9.81], dtype=np.float)

        R = SO3.from_two_directions(d_f, d_t)

        sum_error = 0
        for (x1, x2) in zip(d_t, R * d_f):
            sum_error += np.sqrt((x1 - x2)**2)

        d_t_recovered = R * d_f

        sin_theta = np.cross(d_t, d_t_recovered) / np.linalg.norm(d_t) / np.linalg.norm(d_t_recovered)
        sin_theta = np.linalg.norm(sin_theta)
        self.assertAlmostEqual(sin_theta, 0, default_tol_place)

        # test if determinant is 1
        det = np.linalg.det(R.get_matrix())
        self.assertAlmostEqual(det, 1, default_tol_place)

    def test_inverse(self):
        rot = SO3.from_euler(0.1, -0.2, 0.3)
        tmp = np.dot(rot.get_matrix(), rot.inverse().get_matrix())
        result = test_matrix_equal(tmp, np.eye(3))

        self.assertTrue(result, "Inverse not correct!")

    def test_exp_ln(self):
        rot = SO3()
        so3 = rot.ln()
        result = test_matrix_equal(so3, np.zeros_like(so3))

        self.assertTrue(result, "Log not correct!")

        rot = SO3.from_euler(0.1, -0.2, 0.3)
        so3 = rot.ln()
        result = test_matrix_equal(rot.get_matrix(), SO3.exp(so3).get_matrix())

        self.assertTrue(result)

if (__name__ == "__main__"):
    unittest.main()
