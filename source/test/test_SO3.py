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
