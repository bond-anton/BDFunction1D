import math as m
import numpy as np

from BDFunction1D import Function

import unittest


class TestFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_Function(self):
        f = Function()
        x = m.pi
        self.assertEqual(f.evaluate_point(x), 0.0)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.zeros_like(x))

    def test_new_Function(self):

        class TestF(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        f = TestF()
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x))
