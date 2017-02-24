import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import json

from data_source.interface import DataSourceInterface
from base.noise import *

class SimulatedDeviceInterface:
    """
    """
    def __init__(self):
        pass

    def get_measurement(self):
        pass

class Accelerometer(SimulatedDeviceInterface):
    """
    """
    def __init__(self, noise, gravity_in_world):
        self._noise = noise
        self._gravity_in_world = gravity_in_world

    def get_measurement(self, R_from_body_to_world):
        n = self._noise.get_value()
        m = R_from_body_to_world.inverse() * self._gravity_in_world
        return m + n

class Gyroscope(SimulatedDeviceInterface):
    """
    """
    def __init__(self, noise, bias):
        self._noise = noise
        self._bias = bias

    def get_measurement(self, strapdown_speed):
        n = self._noise.get_value()
        return strapdown_speed + n + self._bias

class Magnetometer(SimulatedDeviceInterface):
    """
    """
    def __init__(self, noise, bias, magnetic_field_in_world):
        self._noise = noise
        self._bias = bias
        self._magnetic_field_in_world = magnetic_field_in_world

    def get_measurement(self, R_from_body_to_world):
        n = self._noise.get_value()
        m = R_from_body_to_world.inverse() * self._magnetic_field_in_world
        return m + n + self._bias

class ConfigParser:
    """
    Simulation config file parser.

    config file (json format) contains:

        gyro noise
        gyro offset

        accel noise
        gravity magnitude

        mag noise
        mag bias
        initial reading

        t, turn rate in body axis

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
        self._turn_rates = []

        with open(filename, 'r') as f:
            object_list = json.load(f)
            print(object_list)
            self._process_json(object_list)

    def get_accelerometer(self):
        return Accelerometer(self._accel_noise, self._gravity)

    def get_gyroscope(self):
        return Gyroscope(self._gyro_noise, self._gyro_bias)

    def get_magnetometer(self):
        return Magnetometer(self._mag_noise, self._mag_bias)

    def get_initial_orientation(self):
        return self._initial_orientation

    def get_turn_rates(self):
        """
        :return a list of dict {"time", "value"}, in ascending order of time
        """
        return self._turn_rates

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
            noise_param = nd["parameter"]
            if ("gaussian" == noise_type):
                noise = GaussianNoise.from_parameter(noise_param)
            elif ("uniform" == noise_type):
                noise = UniformNoise.from_parameter(noise_param)
            else:
                raise RuntimeError("Unknown noise type: {}".format(noise_type))

            result.register_noise(noise, noise_weight)
        print("=== build noise ===")
        print(result)
        return result

    def _process_json(self, object_list):
        """
        """
        for item in object_list:
            item_type = item["type"]
            item_value = item["value"]
            print("item_type is {}".format(item_type))
            print("item_value is {}\n{}".format(type(item_value), item_value))
            if ("accelerometer_noise" == item_type):
                self._accel_noise = self._build_noise(item_value)
            elif ("gyroscope_noise" == item_type):
                self._gyro_noise = self._build_noise(item_value)
            elif ("gyroscope_bias" == item_type):
                self._gyro_bias = item_value
            elif ("magnetometer_noise" == item_type):
                self._mag_noise = self._build_noise(item_value)
            elif ("magnetometer_bias" == item_type):
                self._mag_bias = item_value
            elif ("gravity" == item_type):
                self._gravity = item_value
            elif ("magnetic_field" == item_type):
                self._magnetic_field = item_value
            elif ("initial_orientation" == item_type):
                self._initial_orientation = SO3.exp(item_value)
            elif ("turn_rate" == item_type): # there can be multiple instances
                self._turn_rates.append(item_value)
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
                      (len(self._turn_rates) > 1)

        if (self._ready):
            # sort turn_rates
            pass

class Simulator(DataSourceInterface):
    """
    Simulate 3D rotation based on a provided config file
    """
    def __init__(self, config_filename, update_frequency=1000, report_frequency=100):
        """
        :param initial_orientation of type SO3, rotation from chip to world
        """
        cp = ConfigParser(config_filename)

        if (not cp.ready()):
            raise RuntimeError('Cannot parse config file')

        self._R_from_body_to_world = cp.get_initial_orientation()
        self._accelerometer = cp.get_accelerometer()
        self._gyroscope = cp.get_gyroscope()
        self._magnetometer = cp.get_magnetometer()
        self._turn_rates = cp.get_turn_rates()

        self._time = 0
        self._report_time = 0
        self._current_index = 0 # index in _turn_rates

        self._update_interval = 1 / update_frequency
        self._report_interval = 1 / report_frequency


    def read(self):
        if (self.eof()):
            return None

        self._report_interval += self._report_interval

        while (self._time < self._report_interval):
            turn_rate = self._get_turn_rate(self._time) # speed in old body frame
            if (turn_rate is None):
                return None
            R_from_new_to_old = SO3.exp(turn_rate * self._update_interval)
            self._R_from_body_to_world = self._R_from_body_to_world * R_from_new_to_old
            self._time += self._update_interval

        # generate sensor readings (instantaneous values)
        accel_reading = self._accelerometer.get_measurement(self._R_from_body_to_world)
        gyro_reading = self._accelerometer.get_measurement(turn_rate)
        mag_reading = self._magnetometer.get_measurement(self._R_from_body_to_world)

        result = {"time" : self._report_time,
                  "gyro" : gyro_reading,
                  "accel": accel_reading,
                  "mag" : mag_reading}
        return result

    def eof(self):
        return (self._current_index >= len(self._entries))

    def _get_turn_rate(self, t):
        """
        :retrn turn rate in body frame at timestamp t, or None if eof.
        """
        while ((not eof()) and (self._entries[self._current_index]["time"] < t)):
            self._current_index += 1

        if (eof()):
            return None
        else:
            return self._entries[self._current_index]["value"]

if (__name__ == "__main__"):
    cp = ConfigParser("test_config.json")
