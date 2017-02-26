import numpy as np
import json

class GaussianNoise:
    def __init__(self, **kwargs):
        if ('mu' in kwargs):
            self.mu = np.array(kwargs['mu'], dtype=np.float)
        else:
            self.mu = 0.0;

        if ('sigma' in kwargs):
            self.sigma = np.array(kwargs['sigma'], dtype=np.float)
        else:
            self.sigma = 1.0;

    def __str__(self):
        return '[GaussianNoise] mu = {}, sigma = {}'.format(self.mu, self.sigma)

    def get_value(self):
        return np.random.normal(self.mu, self.sigma)

    @classmethod
    def from_parameters(self, parameters):
        """
        :param parameters [mu, sigma]
        """
        if (len(parameters) != 2):
            raise ValueError("Need 2 parameters.")
        return GaussianNoise(mu=parameters[0], sigma=parameters[1])

class UniformNoise:
    def __init__(self, **kwargs):
        if ('lower' in kwargs):
            self.lower = np.array(kwargs['lower'], dtype=np.float)
        else:
            self.lower = 0.0

        if ('upper' in kwargs):
            self.upper = np.array(kwargs['upper'], dtype=np.float)
        else:
            self.upper = 1.0

    def __str__(self):
        return '[UniformNoise] lower = {}, upper = {}'.format(self.lower, self.upper)

    def get_value(self):
        return np.random.uniform(self.lower, self.upper)

    @classmethod
    def from_parameters(self, parameters):
        """
        :param parameters [lower, upper]
        """
        if (len(parameters) != 2):
            raise ValueError("Need 2 parameters.")
        return UniformNoise(lower=parameters[0], upper=parameters[1])

class CompositeNoise:
    """
    Linear composition of different kinds of noise.
    """
    def __init__(self):
        self._weights = []
        self._noises = []
    
    def __str__(self):
        result = "[CompositeNoise]\n"
        for (w, n) in zip(self._weights, self._noises):
            result = result + "\tweight = {}, {}\n".format(w, str(n))

        return result

    def register_noise(self, noise, weight):
        self._weights.append(weight)
        self._noises.append(noise)

    def get_value(self):
        if (len(self._weights) < 1):
            return None

        values = []
        for (w, n) in zip(self._weights, self._noises):
            values.append(w * n.get_value())
        return sum(values)


if __name__ == '__main__':
    pass

