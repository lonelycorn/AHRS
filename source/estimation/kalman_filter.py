import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from base.SO3 import SO3
from base.SO3 import skew_symmetric_matrix

class KalmanFilterSO3:
    """
    Extended Kalman Filter for estimation of rotations
    Simple version, no consideration of sensor drifting, bias, etc.
    """

    def __init__(self, R0=None, P0=None, gyro_cov=None, acc_cov=None, mag_cov=None,
                 gravity=None, mag_field=None):
        """
        :param R0: Initial mean of the state, 3x3 matrix or SO3
        :param P0: Initial covariance of the state, 3x3 matrix
        :param gyro_cov: 3x3 covariance matrix of the gyro measurement
        :param acc_cov: 3x3 covariance matrix of the accelerometer measurement
        :param mag_cov: 3x3 covariance matrix of the magnetometer measurement
        :param gravity: Gravity in world frame (fixed)
        :param mag_field: Magnetic field in world frame (fixed)
        """
        if (R0 is None):
            self._R_from_world_to_body = SO3()
        elif (isinstance(R0, SO3)):
            self._R_from_world_to_body = R0
        else:
            self._R_from_world_to_body = SO3(R0)

        self._P = P0
        self._P_gyro = gyro_cov
        self._P_acc = acc_cov
        self._P_mag = mag_cov
        self._g = gravity
        self._m0 = mag_field

    def set_initial_pose(self, R0, P0):
        """
        :param R0: Initial mean of the state, 3x3 matrix or SO3
        :param P0: Initial covariance of the state, 3x3 matrix
        """
        if (isinstance(R0, SO3)):
            self._R_from_world_to_body = R0
        else:
            self._R_from_world_to_body = SO3(R0)
        self._P = P0

    def set_sensor_covar(self, gyro_cov, acc_cov, mag_cov):
        """
        :param gyro_cov: 3x3 covariance matrix of the gyro measurement
        :param acc_cov: 3x3 covariance matrix of the accelerometer measurement
        :param mag_cov: 3x3 covariance matrix of the magnetometer measurement
        """
        self._P_gyro = gyro_cov
        self._P_acc = acc_cov
        self._P_mag = mag_cov

    def set_references(self, gravity_ref, mag_ref):
        """
        :param gravity_ref: gravity reference in world/initial frame
        :param mag_ref: magnetic field reference in world/initial frame
        """
        self._g = gravity_ref
        self._m0 = mag_ref

    def get_estimate_mean(self):
        return self._R_from_world_to_body

    def get_estimate_covar(self):
        return self._P

    def get_estimate(self):
        return self._R_from_world_to_body, self._P

    def process_update(self, om, dt, covar=None):
        """
        Update the rotation with gyroscope measurements,
        i.e. angular velocities
        :param om: 1x3 numpy array of angular velocity
        :param dt: time interval
        :param covar: process covar per unit time
        """
        R1 = SO3.exp(-om * dt)
        self._R_from_world_to_body = R1 * self._R_from_world_to_body

        R1 = R1.get_matrix()
        # NOTE: process noise model is 'constant covar per unit time'
        if (covar is None):
            # NOTE: assuming gyro updates at a constant rate
            self._P = self._P_gyro + np.dot(R1, np.dot(self._P, R1))
        else:
            self._P = dt * covar + np.dot(R1, np.dot(self._P, R1))

    def measurement_update(self, J, cov_meas, v):
        """
        Update the mean and covariance based on one measurement
        :param J: Jacobian of the measurement model
        :param cov_meas: measurement covariance
        :param v: difference between measurement and the prediction
        """
        # calculate the Kalman gain
        K = np.dot(np.dot(self._P, J.T),
                   np.linalg.inv(cov_meas + np.dot(J, np.dot(self._P, J.T))))

        # update the mean and covariance with the measurement
        self._R_from_world_to_body = SO3.exp(np.dot(K, v)) * self._R_from_world_to_body
        self._P = np.dot(np.eye(3) - np.dot(K, J), self._P)

    def acc_update(self, acc_meas, acc_covar=None):
        """
        Use accelerometer measurement for update
        Simplest model, ignoring bias and body acceleration
        :param acc_meas: measurement of the acceleration
        :param acc_covar: accelerometer covariance if different from default
        """
        # calculate the predicted acceleration
        acc_pred = self._R_from_world_to_body * self._g

        # calculate Jacobian
        Ja = -skew_symmetric_matrix(acc_pred)

        # perform measurement update
        if acc_covar is None:
            self.measurement_update(Ja, self._P_acc, acc_meas - acc_pred)
        else:
            self.measurement_update(Ja, acc_covar, acc_meas - acc_pred)

    def mag_update(self, mag_meas, mag_covar=None):
        """
        Use magnetometer measurement for update
        Simplest model, ignoring bias, drifting, ant etc.
        :param mag_covar: magnetometer covariance if different from default
        :param mag_meas: measurement of the magnetic field
        """
        # calculate the predicted magnetic field
        mag_pred = self._R_from_world_to_body * self._m0

        # calculate Jacobian
        Jm = -skew_symmetric_matrix(mag_pred)

        # perform measurement update
        if mag_covar is None:
            self.measurement_update(Jm, self._P_mag, mag_meas - mag_pred)
        else:
            self.measurement_update(Jm, mag_covar, mag_meas - mag_pred)


if (__name__ == "__main__"):
    pass
