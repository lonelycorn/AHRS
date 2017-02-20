import numpy

class GaussianNoise:
    def __init__(self, **kwargs):
        if ('mu' in kwargs):
            self.mu = kwargs['mu'] * 1.0
        else:
            self.mu = 0.0;

        if ('sigma' in kwargs):
            self.sigma = kwargs['sigma'] * 1.0
        else:
            self.sigma = 1.0;

    def __str__(self):
        return '[GaussianNoise] mu = %f, sigma = %f' % (self.mu, self.sigma)

    def GetValue(self):
        return numpy.random.normal(self.mu, self.sigma)

class UniformNoise:
    def __init__(self, **kwargs):
        if ('lower' in kwargs):
            self.lower = kwargs['lower'] * 1.0
        else:
            self.lower = 0.0

        if ('upper' in kwargs):
            self.upper = kwargs['upper'] * 1.0
        else:
            self.upper = 1.0

    def __str__(self):
        return '[UniformNoise] lower = %f, upper = %f' % (self.lower, self.upper)

    def GetValue(self):
        return random.uniform(self.lower, self.upper)

class CompositeNoise:
    """
    Linear composition of different kinds of noise.
    """
    def __init__(self):
        raise NotImplementedError('')
    
    def __str__(self):
        raise NotImplementedError('')

    def GetValue(self):
        raise NotImplementedError('')

if __name__ == '__main__':
    pass

