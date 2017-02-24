import numpy as np

class LeastSquareEstimator:
    '''
    A class that performs least square estimation recursively.
    see https://ocw.mit.edu/courses/mechanical-engineering/2-160-identification-estimation-and-learning-spring-2006/lecture-notes/lecture_2.pdf
    '''
    def __init__(self, initial_value, initial_covar, forgetting_factor=1.0):
        '''
        Ctor.
        :param state_size number of state variables to estimate.
        :param forgetting_factor a number within (0.0, 1.0] that discounts old samples.
        '''
        if ((forgetting_factor > 1.0) or \
            (forgetting_factor <= 0.0)):
            raise ValueError('Invalid forgetting factor.')
            
        self._theta = initial_value
        self._P = initial_covar
        self._alpha = forgetting_factor

    def update(self, phi, y):
        '''
        Update the estimate with new sample (phi, y).
        The model is theta.T * phi == y
        '''
        denom = self._alpha + np.dot(phi, np.dot(self._P, phi))

        self._theta += np.dot(self._P, phi) / denom * (y - np.dot(phi, self._theta))
        self._P = 1.0 / self._alpha * (self._P - np.dot(self._P, np.dot(np.outer(phi, phi), self._P)) / denom)

    def get_estimate_mean(self):
        return self._theta

    def get_estimate_covar(self):
        return self._P

if (__name__ == '__main__'):
    pass
