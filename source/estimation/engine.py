import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from base.SO3 import SO3
from base.low_pass_filter import LowPassFilter
from estimation.magnetometer_calibrator import MagnetometerCalibrator
from estimation.kalman_filter import KalmanFilterSO3

class Engine:
    GYRO_NO_MOTION_THRESHOLD = 0.1
    ACCEL_NO_MOTION_THRESHOLD = 9.9
    LOWPASS_GAIN = 0.95

    STATE_INIT = 0
    STATE_CALIBRATE_MOVING = 1 # for mag bias
    STATE_CALIBRATE_STATIC = 2 # for gyro bias, mag ref and accel ref
    STATE_RUNNING = 3

    def __init__(self):
        # FIXME: disabled until we figured out how to use it.
        self._filter = None
        #self._filter = KalmanFilterSO3() # estimates the transform from current chip to initial chip

        self._gyro_lp = LowPassFilter(Engine.LOWPASS_GAIN)
        self._accel_lp = LowPassFilter(Engine.LOWPASS_GAIN)

        self._mag_calibrator = MagnetometerCalibrator(np.zeros(3))
        self._state = Engine.STATE_INIT
        self._mag_ref = None
        self._mag_bias = None
        self._gyro_bias = None
        self._initial_transform = None # transform from initial chip to world

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
            print("[EngineState] INIT")
            # wait until starts to move
            if (not no_motion):
                print("[EngineState] transit to CALIBRATE_MOVING")
                self._state = Engine.STATE_CALIBRATE_MOVING

        elif (self._state == Engine.STATE_CALIBRATE_MOVING):
            print("[EngineState] CALIBRATE_MOVING")
            self._mag_calibrator.update(mag)
            self._mag_bias = self._mag_calibrator.bias
            # wait until found bias, and stopped moving
            if ((self._mag_bias is not None) and \
                (no_motion)):
                print("[EngineState] transit to CALIBRATE_STATIC")
                print("mag bias is {}".format(self._mag_bias))
                self._state = Engine.STATE_CALIBRATE_STATIC

        elif (self._state == Engine.STATE_CALIBRATE_STATIC):
            print("[EngineState] CALIBRATE_STATIC")
            if (no_motion):
                done = self._update_static_calibration(gyro, accel, mag)
                if (done):
                    # TODO: calculate initial transform (roll and pitch) based on accel ref
                    self._state = Engine.STATE_RUNNING
                    
        elif (self._state == Engine.STATE_RUNNING):
            # always do gyro update
            # do accel update iff gravity is dominant
            # do mag update iff mag reading matchs mag param
            pass
        else:
            # invalid state -- should not happen
            assert(false)

    def get_orientation_in_world(self):
        '''
        :return transform from current chip to world.
        '''
        if (self._state < Engine.STATE_RUNNING):
            return None
        return self._initial_transform * self._filter.get_estimate()

    def get_state_string(self):
        """
        :return a string representing the internal state.
        """
        if (self._state == Engine.STATE_INIT):
            return "Init"
        elif (self._state == Engine.STATE_CALIBRATE_MOVING):
            return "Moving calibration (magnetometer)"
        elif (self._state == Engine.STATE_CALIBRATE_STATIC):
            return "Static calibration (gyroscope and accelerometer)"
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
        pass

if (__name__ == '__main__'):
    pass

