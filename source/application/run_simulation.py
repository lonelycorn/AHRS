import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import time
import numpy as np
import threading

from data_source.simulator import Simulator
from estimation.engine import Engine
from visualization.plotter import Plotter, PlotterSharedData


def update_plot(*args):
    """
    *args: plotter, interval
    """
    plotter = args[0]
    interval = args[1]
    
    if (not plotter.stopped):
        plotter.draw()

        threading.Timer(interval, update_plot, plotter, interval).start()

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Estimate 3D rotation in a simulated scenario.')
    parser.add_argument('config_file',
                        type=str,
                        help='simulation config file.')
    parser.add_argument('--update-freq',
                        dest='update_freq',
                        type=int,
                        default=100,
                        action='store',
                        help='update frequency of the simulator; in Hz.')
    parser.add_argument('--sample-freq',
                        dest='sample_freq',
                        type=int,
                        default=10,
                        action='store',
                        help='sample frequency of the simulated sensors; in Hz.')
    parser.add_argument('--plot-freq',
                        dest='plot_freq',
                        type=int,
                        default=5,
                        action='store',
                        help='frequency to update the plot; in Hz.')

    args = parser.parse_args()

    simulator = Simulator(args.config_file, args.update_freq, args.sample_freq)
    estimator = Engine()

    shared_data = PlotterSharedData()
    plotter = Plotter(shared_data)

    interval = 1.0 / args.plot_freq
    last_plot_time = 0
    # we really should use this to draw... but the async update doesn't really draw...
    #update_plot(plotter, interval)

    print("Simulation started.")
    while (not simulator.eof()):
        data = simulator.read()
        t = data["time"]
        gyro = data["gyro"]
        accel = data["accel"]
        mag = data["mag"]

        estimator.update(t, gyro, accel, mag)

        true_orientation = simulator.true_orientation
        print("t = %.2f s, yaw = %.2f deg, pitch = %.2f deg, roll = %.2f deg\n" %\
              (t, true_orientation.get_yaw() * 180.0 / np.pi,
               true_orientation.get_pitch() * 180.0 / np.pi,
               true_orientation.get_roll() * 180.0 / np.pi))

        shared_data.true_orientation = true_orientation
        shared_data.time = simulator.time
        
        # FIXME: using this until we figured out why the async update doesn't work...
        if (simulator.time > last_plot_time + interval):
            plotter.draw()
            input("Press Enter to continue")
            last_plot_time = simulator.time

    print("Simulation ended.")

