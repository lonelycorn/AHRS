import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np

from base.low_pass_filter import LowPassFilter

class TestLowPassFilter(unittest.TestCase):

    VALUE_EQUAL_PLACE = 3 

    def setUp(self):
        pass

    def test_constant_values(self):
        lp = LowPassFilter(0.5)
        values = np.ones(10)

        self.assertIsNone(lp.value)
        for v in values:
            lp.update(v)
            self.assertAlmostEqual(v, lp.value, TestLowPassFilter.VALUE_EQUAL_PLACE)
    
    def test_monotonic_sequence(self):
        lp = LowPassFilter(0.5)
        values = range(5)

        expected_lp_values = [0, 0.5, 1.25, 2.125, 3.0625]
        for (v, ev) in zip(values, expected_lp_values):
            lp.update(v)
            self.assertAlmostEqual(lp.value, ev, TestLowPassFilter.VALUE_EQUAL_PLACE)

if (__name__ == "__main__"):
    unittest.main()
