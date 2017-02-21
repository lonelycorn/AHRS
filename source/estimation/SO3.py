import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from estimation.config import TOLERANCE

def skew_symmetric_matrix(v):
    """
    Get the skew symmetric matrix for a given vector
    :param v: 1D numpy array of the input vector
    :return: 3x3 numpy array of the skew symmetric matrix
    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rodrigues(v):
    """
    Rodrigues' rotation formula.
    See https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    :param v: 1D numpy array, axis-angle representation of the rotation
    :return: 3x3 numpy array of the rotation matrix
    """
    theta = np.linalg.norm(v)
    if theta < TOLERANCE:
        A = 1.0 - theta**2 / 6.0
        B = 0.5 - theta**2 / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta**2

    K = skew_symmetric_matrix(v)
    return np.eye(3) + np.dot(A, K) + np.dot(B, np.dot(K, K))


class SO3:
    """
    Special Orthogonal Group in 3D space.
    This is a generic representation for rotations in 3D space.
    """
    def __init__(self, R=None):
        """
        Constructor with an optional initial rotation matrix
        :param R: 3x3 numpy array rotation matrix used for initialization
        """
        if R is None:
            self._R = np.eye(3)
        else:
            # TODO: check and normalize R
            self._R = R

    def __str__(self):
        """
        For printing
        :return: string representation of the object
        """
        return np.array_str(self._R)

    def __mul__(self, other):
        """
        Overloading the multiplication operator,
        work either on another SO3 or a vector
        :param other: can be either an SO3 or a 1x3 numpy array
        :return: same type as the input
        """
        if isinstance(other, self.__class__):
            R1 = self._R
            R2 = other.get_matrix()
            return SO3(R1.dot(R2))
        else:
            return np.dot(self._R, other)

    @classmethod
    def from_euler(cls, roll, pitch, yaw):
        """
        Constructor using Tait-Bryan angles/euler angles
        Sequence of rotation: z-y-x'', i.e. yaw, pitch, roll
        :param yaw: yaw angle in radian
        :param pitch: pitch angle in radian
        :param roll: roll angle in radian
        :return: SO3 object constructed from yaw, pitch, roll
        """
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(roll), -np.sin(roll)],
                       [0.0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0.0, np.sin(pitch)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(pitch), 0.0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                       [np.sin(yaw), np.cos(yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        return SO3(R=np.dot(Rz, np.dot(Ry, Rx)))

    @classmethod
    def from_so3(cls, so3):
        """
        Constructor using so3/axis-angle representation
        :param so3: 1x3 numpy array of the axis-angle representation
        :return: SO3 object from the so3/axis-angle representation
        """
        return SO3(R=rodrigues(so3))

    @classmethod
    def exp(cls, so3):
        """
        Convert axis-angle representation to SO3
        This is the inverse operatin of ln()
        :param so3: 1x3 numpy array of so3
        :return: SO3 converted from so3
        """
        return SO3.from_so3(so3)

    def ln(self):
        """
        Get the axis-angle representation of the SO3 object
        :return: 1x3 numpy array of the axis-angle
        """
        R = self._R
        cos_theta = 0.5 * (np.trace(R) - 1.0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

        if theta < 1e-5:
            A = (1.0 + theta**2 / 6.0) * 0.5
        else:
            A = 0.5 * theta / np.sin(theta)

        return v * A

    def get_matrix(self):
        """
        Method to get the rotation matrix from SO3 object
        :return: 3x3 numpy array of the rotation matrix
        """
        return self._R

    def inverse(self):
        """
        Method to get the inverse of the transformation
        :return: SO3 object that has the inverted rotation matrix
        """
        R = self._R.T
        return SO3(R)

    def rectify(self):
        """
        Make sure this is a valid SO3, i.e. orthonormal
        :return: no return
        """
        u0 = self._R[0, :]
        u0 /= np.linalg.norm(u0)

        u1 = self._R[1, :]
        u1 -= np.dot(u0, u1) * u0

        u2 = np.cross(u0, u1)

        self._R = np.vstack((u0, u1, u2))

    def reset(self):
        """
        Reset to identity
        :return: no return
        """
        self._R = np.eye(3)

    def adjoint(self):
        """
        Get the adjoint matrix
        :return: 3x3 numpy array of the adjoint matrix
        """
        return self._R

    def get_roll(self):
        """
        Get the roll angle from the rotation matrix
        :return: float number of the roll angle in radian
        """
        return np.arctan2(self._R[2, 1], self._R[2, 2])

    def get_pitch(self):
        """
        Get the pitch angle from the rotation matrix
        :return: float number of the pitch angle in radian
        """
        return np.arcsin(-self._R[2, 0])

    def get_yaw(self):
        """
        Get the yaw angle from the rotation matrix
        :return: float number of the yaw angle in radian
        """
        return np.arctan2(self._R[1, 0], self._R[0, 0])


if (__name__ == "__main__"):
    pass
