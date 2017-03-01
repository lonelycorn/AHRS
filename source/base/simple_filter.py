import numpy as np

class AverageFilter:
    def __init__(self):
        self._sample_sum = None
        self._sample_count = 0

    def reset(self):
        self._sample_sum = None
        self._sample_count = 0

    def update(self, x):
        v = np.array(x, dtype=np.float)
        if (self._sample_sum is None):
            self._sample_sum = v
            self._sample_count = 1
        else:
            self._sample_sum += v
            self._sample_count += 1

    @property
    def value(self):
        if (self._sample_count < 1):
            return None
        else:
            return self._sample_sum / self._sample_count

    @property
    def count(self):
        return self._sample_count

class LowPassFilter:
    def __init__(self, low_pass_gain):
        """
        """
        if ((low_pass_gain <= 0.0) or (low_pass_gain >= 1.0)):
            raise ValueError("Invalid gain: %f" % (low_pass_gain * 1.0))
        self._alpha = low_pass_gain

        self._value = None

    def reset(self):
        self._value = None

    def update(self, x):
        v = np.array(x, dtype=np.float)
        if (self._value is None):
            self._value = v
        else:
            self._value = self._value * self._alpha + v * (1.0 - self._alpha)

    @property
    def value(self):
        return self._value

if (__name__ == "__main__"):
    pasa
