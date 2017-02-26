import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import numpy as np

from data_source.simulator import Simulator
from estimation.engine import Engine


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Estimate 3D rotation in a simulated scenario.')
    parser.add_argument('config_file',
                        type=str,
                        help='simulation config file.')

    args = parser.parse_args()

    simulator = Simulator(args.config_file, 10, 2)
    estimator = Engine()

    print("Simulation started.")
    while (not simulator.eof()):
        data = simulator.read()
        t = data["time"]
        gyro = data["gyro"]
        accel = data["accel"]
        mag = data["mag"]
        true_orientation = simulator.get_true_orientation()
        print("t = %.2f s, yaw = %.2f deg, pitch = %.2f deg, roll = %.2f deg\n" %\
              (t, true_orientation.get_yaw() * 180.0 / np.pi,
               true_orientation.get_pitch() * 180.0 / np.pi,
               true_orientation.get_roll() * 180.0 / np.pi))

    print("Simulation ended.")


