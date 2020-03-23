import math as m
import numpy as np

from BDFunction1D import Function, Functional

import unittest


class TestFunctional(unittest.TestCase):

    def setUp(self):
        pass

    def test_Functional(self):
        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        class TestF2(Function):
            def evaluate_point(self, x):
                return m.sin(x)

        class TestFunctional1(Functional):
            def __init__(self, f):
                super(TestFunctional1, self).__init__(f)

        f = TestFunctional1(TestF1())
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))
        f.f = TestF2()
        np.testing.assert_allclose(f.evaluate(x), np.sin(x))
