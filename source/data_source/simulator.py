import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import json
import copy

from data_source.interface import DataSourceInterface
from base.noise import *
from base.SO3 import SO3

class SimulatedDeviceInterface:
    """
    """
    def __init__(self):
        pass

    def get_measurement(self):
        pass

class Accelerometer(SimulatedDeviceInterface):
    """
    Simulated strapdown accelerometer.
    """
    def __init__(self, noise, gravity_in_world):
        """
        :param noise
        :param gravity_in_world gravity measurement in world ref frame.
        """
        self._noise = noise
        self._gravity_in_world = gravity_in_world

    def get_measurement(self, R_from_body_to_world):
        n = self._noise.get_value()
        m = R_from_body_to_world.inverse() * self._gravity_in_world
        # as d'Alembert put it, acceleration is in opposite direction of inertial force
        return - m + n

class Gyroscope(SimulatedDeviceInterface):
    """
    Simulated strapdown gyroscope.
    """
    def __init__(self, noise, bias):
        self._noise = noise
        self._bias = bias

    def get_measurement(self, strapdown_speed):
        """
        :param strapdown_speed turn rate in body ref frame.
        """
        n = self._noise.get_value()
        return strapdown_speed + n + self._bias

class Magnetometer(SimulatedDeviceInterface):
    """
    Simulated strapdown magnetometer.
    """
    def __init__(self, noise, bias, magnetic_field_in_world):
        """
        :param magnetic_field_in_world earth's magnetic field in world ref frame.
        """
        self._noise = noise
        self._bias = bias
        self._magnetic_field_in_world = magnetic_field_in_world

    def get_measurement(self, R_from_body_to_world):
        n = self._noise.get_value()
        m = R_from_body_to_world.inverse() * self._magnetic_field_in_world
        return m + n + self._bias

class ConfigParser:
    """
    Parser for simulation config files.
    """
    def __init__(self, filename):
        self._ready = False

        self._accel_noise = None
        self._gyro_noise = None
        self._gyro_bias = None
        self._mag_noise = None
        self._mag_bias = None
        self._gravity = None
        self._magnetic_field = None
        self._initial_orientation = None
        self._turn_rates = None

        with open(filename, 'r') as f:
            json_object = json.load(f)
            self._process_json_object(json_object)

    def get_accelerometer(self):
        return Accelerometer(self._accel_noise, self._gravity)

    def get_gyroscope(self):
        return Gyroscope(self._gyro_noise, self._gyro_bias)

    def get_magnetometer(self):
        return Magnetometer(self._mag_noise, self._mag_bias, self._magnetic_field)

    def get_initial_orientation(self):
        # return a copy to avoid any changes
        return copy.deepcopy(self._initial_orientation)

    def get_turn_rates(self):
        """
        :return a list of dict {"time", "value"}, in ascending order of time
        """
        # return a copy to avoid any changes
        return copy.deepcopy(self._turn_rates)

    def ready(self):
        return self._ready

    def _build_noise(self, noise_descriptions):
        """
        :param noise_descriptions a list of dict
        """
        result = CompositeNoise()

        for nd in noise_descriptions:
            noise_type = nd["type"]
            noise_weight = nd["weight"]
            noise_param = nd["parameters"]
            if ("gaussian" == noise_type):
                noise = GaussianNoise.from_parameters(noise_param)
            elif ("uniform" == noise_type):
                noise = UniformNoise.from_parameters(noise_param)
            else:
                raise RuntimeError("Unknown noise type: {}".format(noise_type))

            result.register_noise(noise, noise_weight)

        return result

    def _process_json_object(self, json_object):
        """
        :param  json_object
        """
        for item in json_object:
            item_type = item["type"]
            item_value = item["value"]

            if ("accelerometer_noise" == item_type):
                self._accel_noise = self._build_noise(item_value)
            elif ("gyroscope_noise" == item_type):
                self._gyro_noise = self._build_noise(item_value)
            elif ("gyroscope_bias" == item_type):
                self._gyro_bias = np.array(item_value)
            elif ("magnetometer_noise" == item_type):
                self._mag_noise = self._build_noise(item_value)
            elif ("magnetometer_bias" == item_type):
                self._mag_bias = np.array(item_value)
            elif ("gravity" == item_type):
                self._gravity = np.array(item_value)
            elif ("magnetic_field" == item_type):
                self._magnetic_field = np.array(item_value)
            elif ("initial_orientation" == item_type):
                self._initial_orientation = SO3.exp(item_value)
            elif ("turn_rates" == item_type): # there can be multiple instances
                self._turn_rates = item_value
                for tr in self._turn_rates:
                    tr["time"] = np.float(tr["time"])
                    tr["value"] = np.array(tr["value"])
            else:
                raise RuntimeError("Unknown json object type: {}".format(item_type))


        self._ready = (self._accel_noise is not None) and \
                      (self._gyro_noise is not None) and \
                      (self._gyro_bias is not None) and \
                      (self._mag_noise is not None) and \
                      (self._mag_bias is not None) and \
                      (self._gravity is not None) and \
                      (self._magnetic_field is not None) and \
                      (self._initial_orientation is not None) and \
                      (self._turn_rates is not None) and \
                      (len(self._turn_rates) > 1)

        if (self._ready):
            # make sure the entries are sorted in ascending order of time
            n = len(self._turn_rates)
            for i in range(0, n-1):
                for j in range(i+1, n):
                    if (self._turn_rates[i]["time"] > self._turn_rates[j]["time"]):
                        tmp = self._turn_rates[i]
                        self._turn_rates[i] = self._turn_rates[j]
                        self._turn_rates[j] = tmp

class Simulator(DataSourceInterface):
    """
    Simulate 3D rotation based on a provided config file
    """
    def __init__(self, config_filename, update_frequency=1000, report_frequency=100):
        """
        :param config_filename  URI to the simulation config file.
        :param update_frequency frequency to update the internal kinematics; in Hz.
        :param report_frequency frequency to report sensor measurements; in Hz.
        """
        cp = ConfigParser(config_filename)

        if (not cp.ready()):
            raise RuntimeError('Cannot parse config file: {}'.format(config_filename))

        self._R_from_body_to_world = cp.get_initial_orientation()
        self._accelerometer = cp.get_accelerometer()
        self._gyroscope = cp.get_gyroscope()
        self._magnetometer = cp.get_magnetometer()
        self._turn_rates = cp.get_turn_rates()
        self._current_index = 0 # index in self._turn_rates

        self._time = 0
        self._report_time = 0

        self._update_interval = 1.0 / update_frequency
        self._report_interval = 1.0 / report_frequency

    def read(self):
        if (self.eof()):
            return None

        self._report_time += self._report_interval

        stop_threshold = 0.5 * self._update_interval
        while (np.abs(self._time  - self._report_time) > stop_threshold):
            turn_rate = self._get_turn_rate(self._time) # speed in old body frame
            if (turn_rate is None):
                return None
            R_from_new_to_old = SO3.exp(turn_rate * self._update_interval)
            self._R_from_body_to_world = self._R_from_body_to_world * R_from_new_to_old
            self._time += self._update_interval

        # generate sensor readings (instantaneous values)
        accel_reading = self._accelerometer.get_measurement(self._R_from_body_to_world)
        gyro_reading = self._gyroscope.get_measurement(turn_rate)
        mag_reading = self._magnetometer.get_measurement(self._R_from_body_to_world)

        result = {"time" : self._report_time,
                  "gyro" : gyro_reading,
                  "accel": accel_reading,
                  "mag" : mag_reading}
        return result

    def eof(self):
        # we need to check if there is enough data for the next read
        return (self._current_index + 1 >= len(self._turn_rates))

    def get_true_orientation(self):
        """
        :return rotation from body to world, in SO3 representation.
        """
        return copy.deepcopy(self._R_from_body_to_world)

    def _get_turn_rate(self, t):
        """
        :retrn turn rate in body frame at timestamp t, or None if eof.
        """
        stop_threshold = 0.5 * self._update_interval
        while ((self._current_index + 1 < len(self._turn_rates)) and \
               (self._turn_rates[self._current_index + 1]["time"] <= t - stop_threshold)):
            self._current_index += 1

        if (self._current_index < len(self._turn_rates)):
            # there is still data
            return self._turn_rates[self._current_index]["value"]
        else:
            return None

if (__name__ == "__main__"):
    pass
