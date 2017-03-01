import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np

from base.simple_filter import LowPassFilter, AverageFilter

class TestSimpleFilter(unittest.TestCase):

    VALUE_EQUAL_PLACE = 3 

    def setUp(self):
        pass

    def test_low_pass_filter_constant_values(self):
        lp = LowPassFilter(0.5)
        values = np.ones(10)

        self.assertIsNone(lp.value)
        for v in values:
            lp.update(v)
            self.assertAlmostEqual(v, lp.value, TestSimpleFilter.VALUE_EQUAL_PLACE)
    
    def test_low_pass_filter_monotonic_sequence(self):
        lp = LowPassFilter(0.5)
        values = range(5)

        expected_lp_values = [0, 0.5, 1.25, 2.125, 3.0625]
        for (v, ev) in zip(values, expected_lp_values):
            lp.update(v)
            self.assertAlmostEqual(lp.value, ev, TestSimpleFilter.VALUE_EQUAL_PLACE)

    def test_average_filter_constant_values(self):
        a = AverageFilter()
        values = np.ones(10)

        self.assertIsNone(a.value)
        for (i, v) in enumerate(values):
            a.update(v)
            self.assertAlmostEqual(v, a.value, TestSimpleFilter.VALUE_EQUAL_PLACE)
            self.assertEqual(a.count, i + 1)

    def test_average_filter_monotonic_sequence(self):
        a = AverageFilter()
        values = range(5)

        expected_a_values = [0, 0.5, 1.0, 1.5, 2.0]
        for (v, ev) in zip(values, expected_a_values):
            a.update(v)
            self.assertAlmostEqual(a.value, ev, TestSimpleFilter.VALUE_EQUAL_PLACE)



if (__name__ == "__main__"):
    unittest.main()
