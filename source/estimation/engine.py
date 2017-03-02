import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from base.SO3 import SO3
from base.simple_filter import LowPassFilter, AverageFilter
from estimation.magnetometer_calibrator import MagnetometerCalibrator
from estimation.kalman_filter import KalmanFilterSO3

class Engine:
    GYRO_NO_MOTION_THRESHOLD = 0.1
    ACCEL_NO_MOTION_THRESHOLD = 10.0 # FIXME we may need a bigger value
    LOWPASS_GAIN = 0.9
    STATIC_CAL_SAMPLE_COUNT = 200
    SENSOR_COVAR_AMPLIFIER = 2.0 # covar obtained after static calibration would be amplified for better stability
    INITIAL_POSE_COVAR = 1e1 # diagonal

    STATE_INIT = 0
    STATE_CALIBRATE_MOVING = 1 # for mag bias
    STATE_CALIBRATE_STATIC = 2 # for gyro bias, mag ref and accel ref
    STATE_RUNNING = 3

    def __init__(self):
        self._filter = KalmanFilterSO3() # estimates the transform from current chip to initial chip

        self._gyro_lp = LowPassFilter(Engine.LOWPASS_GAIN)
        self._accel_lp = LowPassFilter(Engine.LOWPASS_GAIN)

        self._gyro_avg = AverageFilter()
        self._accel_avg = AverageFilter()
        self._mag_avg = AverageFilter()

        self._mag_calibrator = MagnetometerCalibrator(np.zeros(3))

        self._state = Engine.STATE_INIT
        self._last_update_time = 0.0

    def set_mag_param(self, mag_bias):
        '''
        update the mag parameters.
        Could be used as a hacky way to advance the internal state machine,
        but only in the simulation.
        '''
        if (self._state < Engine.STATE_CALIBRATE_STATIC):
            self._state = Engine.STATE_CALIBRATE_STATIC
        self._mag_bias = mag_bias

    def update(self, t, gyro, accel, mag):
        """
        """
        t *= 1.0
        gyro = np.array(gyro, dtype=np.float)
        accel = np.array(accel, dtype=np.float)
        mag = np.array(mag, dtype=np.float)

        # update low pass filters
        self._gyro_lp.update(gyro)
        self._accel_lp.update(accel)

        no_motion = self._check_no_motion(gyro, accel)

        if (self._state == Engine.STATE_INIT):
            # wait until starts to move
            if (not no_motion):
                print("[EngineState] transit from INIT to CALIBRATE_MOVING")
                self._state = Engine.STATE_CALIBRATE_MOVING

        elif (self._state == Engine.STATE_CALIBRATE_MOVING):
            self._mag_calibrator.update(mag)
            self._mag_bias = self._mag_calibrator.bias
            # wait until found bias, and stopped moving
            if ((self._mag_bias is not None) and \
                (no_motion)):
                print("[EngineState] transit from CALIBRATE_MOVING to CALIBRATE_STATIC")
                print("mag bias is {}".format(self._mag_bias))
                self._state = Engine.STATE_CALIBRATE_STATIC

        elif (self._state == Engine.STATE_CALIBRATE_STATIC):
            if (no_motion): # only update when device is stationary
                done = self._update_static_calibration(gyro, accel, mag)
                if (done):
                    # NOTE: acceleration is in the opposite direction of the corresponding inertial force
                    gravity_in_body = self._accel_avg.value
                    gravity_in_world = np.array([0, 0, 1], dtype=np.float) * np.linalg.norm(gravity_in_body)
                    R_from_body_to_world = SO3.from_two_directions(gravity_in_body, gravity_in_world)
                    initial_pose_covar = np.eye(3) * Engine.INITIAL_POSE_COVAR

                    gyro_bias = self._gyro_avg.value
                    gyro_covar = self._gyro_avg.covar * Engine.SENSOR_COVAR_AMPLIFIER

                    accel_covar = self._accel_avg.covar * Engine.SENSOR_COVAR_AMPLIFIER
                    
                    mag_ref = R_from_body_to_world.inverse() * self._mag_avg.value
                    mag_covar = self._mag_avg.covar * Engine.SENSOR_COVAR_AMPLIFIER

                    # initialize the kalman filter here.
                    self._filter.set_initial_pose(R_from_body_to_world, initial_pose_covar)
                    self._filter.set_sensor_covar(gyro_covar, accel_covar, mag_covar)
                    self._filter.set_references(gravity_in_world, mag_ref)

                    self._state = Engine.STATE_RUNNING

                    print("[EngineState] transit from CALIBRATE_STATIC to RUNNING")
                    print("initial orientation = {}\nroll = {}, pitch = {}, yaw = {}".format(
                        R_from_body_to_world.ln(), R_from_body_to_world.get_roll(),
                        R_from_body_to_world.get_pitch(), R_from_body_to_world.get_yaw()))
                    print("gravity in world = {}".format(gravity_in_world))
                    print("gyro bias = {}".format(gyro_bias))
                    print("gyro covar = \n{}".format(gyro_covar))
                    print("accel covar = \n{}".format(accel_covar))
                    print("mag ref = {}".format(mag_ref))
                    print("mag covar = {}".format(mag_covar))
                    

        elif (self._state == Engine.STATE_RUNNING):
            dt = t - self._last_update_time

            # always do gyro update
            self._filter.process_update(gyro, dt)

            # do accel update iff gravity is dominant
            if (np.linalg.norm(accel) < Engine.ACCEL_NO_MOTION_THRESHOLD):
                self._filter.acc_update(accel)

            # do mag update iff mag reading matchs mag param
            mag_calibrated = self._mag_calibrator.calibrate_measurement(mag)
            if (mag_calibrated is not None):
                self._filter.mag_update(mag_calibrated)
                    
        else:
            # invalid state -- should not happen
            assert(False)

        self._last_update_time = t

    def get_orientation_in_world(self):
        '''
        :return transform from current chip to world.
        '''
        if (self._state < Engine.STATE_RUNNING):
            return None
        return self._filter.get_estimate_mean().inverse()

    def get_state_string(self):
        """
        :return a string representing the internal state.
        """
        if (self._state == Engine.STATE_INIT):
            return "Init"
        elif (self._state == Engine.STATE_CALIBRATE_MOVING):
            return "Moving calibration"
        elif (self._state == Engine.STATE_CALIBRATE_STATIC):
            return "Static calibration"
        elif (self._state == Engine.STATE_RUNNING):
            return "Running"
        else:
            raise RuntimeError("Invalid state: {}".format(self._state))

    def _check_no_motion(self, gyro, accel):
        """
        :return True if the barely moving
        """
        tg = Engine.GYRO_NO_MOTION_THRESHOLD
        ta = Engine.ACCEL_NO_MOTION_THRESHOLD
        # trivial motion both instantaneously and recently
        return ((np.linalg.norm(gyro) < tg) and \
                (np.linalg.norm(self._gyro_lp.value) < tg) and \
                (np.linalg.norm(accel) < ta) and \
                (np.linalg.norm(self._accel_lp.value) < ta))

    def _update_static_calibration(self, gyro, accel, mag):
        """
        estimate gyro offset, mag ref and accel ref
        :return True if finished.
        """
        self._gyro_avg.update(gyro)
        self._accel_avg.update(accel)
        self._mag_avg.update(mag - self._mag_bias)

        return ((self._gyro_avg.count > Engine.STATIC_CAL_SAMPLE_COUNT) and \
                (self._accel_avg.count > Engine.STATIC_CAL_SAMPLE_COUNT) and \
                (self._mag_avg.count > Engine.STATIC_CAL_SAMPLE_COUNT))

if (__name__ == '__main__'):
    pass

