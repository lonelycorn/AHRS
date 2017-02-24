import numpy as np

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

