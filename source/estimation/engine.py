import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from estimation.kalman_filter import KalmanFilter
from estimation.SO3 import SO3

class Engine:
    STATE_INIT = 0
    STATE_CALIBRATE_MOVING = 1 # for mag bias
    STATE_CALIBRATE_STATIC = 2 # for gyro bias, mag ref and accel ref
    STATE_RUNNING = 2

    def __init__(self):
        self._filter = KalmanFilter() # estimates the transform from current chip to initial chip
        self._state = ENGINE_STATE_INIT
        self._mag_ref = None
        self._mag_bias = None
        self._gyro_bias = None
        self._initial_transform = None # transform from initial chip to world

    def set_mag_param(self, mag_bias, mag_intensity):
        '''
        update the mag parameters.
        Could be used as a hacky way to advance the internal state machine,
        but only in the simulation.
        '''
        if (self._state < Engine.STATE_CALIBRATE_STATIC)
            self._state = Engine.STATE_CALIBRATE_STATIC
        self._mag_bias = mag_bias
        self._mag_intensity = mag_intensity

    def update(self, t, gyro, accel, mag):
        if (self._state == Engine.STATE_INIT):
            # wait until starts to move
            pass
        elif (self._state == Engine.STATE_CALIBRATE_MOVING):
            # estimate mag bias and mag intensity
            pass
        elif (self._state == Engine.STATE_CALIBRATE_STATIC):
            # estimate gyro offset, mag ref and accel ref
            # calculate initial transform (roll and pitch) based on accel ref
            pass
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

if (__name__ == '__main__'):
    pass

