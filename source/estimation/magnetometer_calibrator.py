import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from base.least_square_estimator import LeastSquareEstimator

class MagnetometerCalibrator:
    """
    Online calibration of the hard-iron and the soft-iron effects.
    FIXME: using a sphere model now, which produces the hard-iron calibration only.
           we may need to upgrade to an ellipsoid model to include the soft-iron
           calibration.
    """
    MIN_SAMPLE_COUNT = 24
    COARSE_FACTOR = 4
    INITIAL_COVAR = 1e3
    ERROR_MARGIN_RATIO = 0.1

    def __init__(self, center_guess):
        """
        :param center_guess: 3-by-1 numpy array; initial guess for the static bias
        :param radius_guess: initial guess for the magnetic field intensity
        """
        self._samples = {}
        self._coarse_samples = {}
        initial_guess = np.array([center_guess[0],
                                  center_guess[1],
                                  center_guess[2], 
                                  0.0], dtype=np.float)
        initial_covar = np.eye(4) * MagnetometerCalibrator.INITIAL_COVAR
        self._estimator = LeastSquareEstimator(initial_guess, initial_covar)

    def update(self, mag):
        """
        Update the mag bias estimate with the new raw measurement.
        :param mag: 3-by-1 numpy array; raw measurement from the magnetometer.
        """
        self._add_sample(mag)
        (phi, y) = self._convert_to_estimator_sample(mag)        
        self._estimator.update(phi, y)

        #estimate = self._estimator.get_estimate_mean()
        #print("[MagCal] center={}, sample size={}, coarse sample size={}".format(\
        #        estimate[0:3], len(self._samples), len(self._coarse_samples)))


    @property
    def bias(self):
        """
        :return static bias of the magnetometer; or None if not yet available.
        """
        if (not self._ready()):
            return None
        else:
            estimate = self._estimator.get_estimate_mean()
            center = np.array(estimate[0:3], dtype=np.float)
            return center

    @property
    def intensity(self):
        """
        :return intensity of the external magnetic field; or None if not yet available.
        """
        if (not self._ready()):
            return None
        else:
            estimate = self._estimator.get_estimate_mean()
            center = np.array(estimate[0:3], dtype=np.float)
            radius = np.sqrt(estimate[3]**2 + np.dot(center, center))
            return radius

    def calibrate_measurement(self, mag, error_margin_ratio=None):
        """
        :return the calibrated measurement, or None if not yet calibrated, or the raw
                measurement does not fit into the model within the specified error margin.
        """
        if (not self._ready()):
            return None 

        if (error_margin_ratio is None):
            error_margin_ratio = MagnetometerCalibrator.ERROR_MARGIN_RATIO

        center = self.bias
        radius = self.intensity
        calibrated = mag - center

        error = np.abs(radius - np.linalg.norm(calibrated))
        if (error < error_margin_ratio * radius):
            return calibrated
        else:
            # possibly a corrupted measurement
            return None

    def _convert_to_estimator_sample(self, mag):
        """
        convert raw magnetometer measurements to estimator samples
        :return (phi, y) the sample input and the sample output.
        """
        # a sphere centered at [x_c, y_c, z_c] with radius R is described by
        # (x - x_c)**2 + (y - y_c)**2 + (z - z_c)**2 == R**2
        # rearranging, we have
        # [2*x, 2*y, 2*z, 1] * [x_c, y_c, z_c, d].T == x**2 + y**2 + z**2
        # where d = R**2 - x_c**2 - y_c**2 - z_c**2
        
        phi = np.array([mag[0] * 2.0,
                        mag[1] * 2.0,
                        mag[2] * 2.0,
                        1.0], dtype=np.float)
        y = np.dot(mag, mag)
        return (phi, y)

    def _ready(self):
        k = MagnetometerCalibrator.MIN_SAMPLE_COUNT
        return ((len(self._samples) > k) and \
                (len(self._coarse_samples) > k))

    def _add_sample(self, mag):
        """
        Add the provided sample to sample history
        """
        hash_value = self._get_sample_hash(mag)
        self._samples[hash_value] = mag

        hash_value = self._get_coarse_sample_hash(mag)
        self._coarse_samples[hash_value] = mag

    def _get_sample_hash(self, mag):
        """
        Calculate the hash value for the given mag value.
        :param mag: 3-by-1 numpy array.
        """
        x = np.int(np.round(mag[0]))
        y = np.int(np.round(mag[1]))
        z = np.int(np.round(mag[2]))
        return x ^ y ^ z
    
    def _get_coarse_sample_hash(self, mag):
        """
        similar to _get_sample_hash, but uses a larger bin
        """
        k = 1.0 / MagnetometerCalibrator.COARSE_FACTOR
        coarse_mag = [mag[0] * k, mag[1] * k, mag[2] * k]
        return self._get_sample_hash(coarse_mag)


if (__name__ == "__main__"):
    pass
