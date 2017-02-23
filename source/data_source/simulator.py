import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import json

from data_source.interface import Interface

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
        self._accelerometer = None
        self._gyroscope = None
        self._magnetometer = None
        self._initial_orientation = None
        self._entries = None
        self._ready = False

        with open(filename, 'r') as f:
            object_list = json.load(f)
            print(json_object)
            self._process_json()

    def get_accelerometer(self):
        return self._accelerometer

    def get_gyroscope(self):
        return self._gyroscope

    def get_magnetometer(self):
        return self._magnetometer

    def get_initial_orientation(self):
        return self._initial_orientation

    def get_turn_rates(self):
        """
        :return a list of dict {"time", "value"}, in ascending order of time
        """
        return self._entries

    def ready(self):
        return self._ready

    def _process_json(self, object_list):
        """
        """
        for item in object_list:
            item_type = item["type"]
            if ("accelerometer" == item_type):
                pass
            elif ("gyroscope" == item_type):
                pass
            elif ("magnetometer" == item_type):
                pass
            elif ("gravity" == item_type):
                pass
            elif ("magnetic_field" == item_type):
                pass
            elif ("initial_orientation" == item_type):
                pass
            elif ("turn_rate" == item_type):
                pass
            else:
                raise RuntimeError("Unknown type: {}".format(item_type))


        self._ready = (self._accelerometer is not None) and \
                      (self._gyroscope is not None) and \
                      (self._magnetometer is not None) and \
                      (self._initial_orientation is not None) and \
                      (self._turn_rates is not None) and \
                      (len(self._turn_rates) > 1)

        if (self._ready):
            # sort turn_rates
            pass
        else:
            self._accelerometer = None
            self._gyroscope = None
            self._magnetometer = None
            self._initial_orientation = None
            self._turn_rates = None


class Simulator(DataSourceInterface):
    """
    Simulate 3D rotation based on a provided config file
    """
    def __init__(self, config_filename, update_frequency=1000, report_frequency=100):
        """
        :param initial_orientation of type SO3, rotation from chip to world
        """
        cp = ConfigParser(config_filename)

        if (not cp.ready())
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
    pass
