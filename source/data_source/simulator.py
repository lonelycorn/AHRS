import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy

from data_source.interface import Interface

class ConfigReader:
    """
    Simulation config file reader.

    log format:

        gyro noise
        gyro offset

        accel noise
        gravity magnitude

        mag noise
        mag bias
        initial reading

        N = number of entries
        t_k, rotation speed in body axis
        
    """
    def __init__(self, log_filename):
        pass

class Simulator(Interface):
    """
    Simulate 3D rotation based on a provided config file
    """
    def __init__(self):
        pass

    def Read(self):
        pass

if (__name__ == "__main__"):
    pass
