import numpy as np

class LowPassFilter:
    def __init__(self, low_pass_gain):
        """
        """
        if ((low_pass_gain <= 0.0) or (low_pass_gain >= 1.0)):
            raise ValueError("Invalid gain: %f" % (low_pass_gain * 1.0))
        self._alpha = low_pass_gain

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
