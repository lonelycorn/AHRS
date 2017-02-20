import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse

from data_source.simulator import Simulator
from estimation.engine import Engine



if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Estimate 3D rotation on a simulated object.')
    parser.add_argument('config_file', dest='config_file', action='store_const',
                        help='simulation config file')

    args = parser.parse_args()

    simulator = Simulator(args.config_file)
    estimator = Engine()

    while (not simulator.EOF()):
        t, gyro, accel, mag = simulator.Read()

        estimator.update(t, gyro, accel, mag)


