import unittest
from SO3 import SO3
import numpy as np

default_tol_place = 2


def test_matrix_equal(m1, m2, tol=0.001):
    """
    Check if two matrices are the same
    :param m1: 2D numpy array of the first matrix
    :param m2: 2D numpy array of the second matrix
    :param tol: tolerance in number of decimal places
    :return: True if two matrices equal, False otherwise
    """
    if not m1.shape == m2.shape:
        return False

    for x, y in np.nditer([m1, m2]):
        if np.abs(x - y) > tol:
            return False

    return True


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
