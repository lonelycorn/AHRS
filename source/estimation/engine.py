import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from estimation.kalman_filter import KalmanFilter

class Engine:
    ENGINE_STATE_INIT = 0
    ENGINE_STATE_CALIBRATING = 1
    ENGINE_STATE_RUNNING = 2

    def __init__(self):
        self._filter = KalmanFilter()
        self._state = ENGINE_STATE_INIT

    def SetGyroOffset(self, offset):
        pass

    def SetMagReference(self, ref):
        pass

    def Update(self, t, gyro, accel, mag):
        pass

if (__name__ == '__main__'):
    pass

