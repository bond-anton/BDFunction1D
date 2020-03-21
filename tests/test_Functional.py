import numpy as np

from BDFunction1D import Function, Functional

import unittest


class TestFunctional(unittest.TestCase):

    def setUp(self):
        pass

    def test_Functional(self):
        class test_F1(Function):
            def evaluate(self, x):
                return np.sqrt(x)

        class test_F2(Function):
            def evaluate(self, x):
                return np.sin(x)

        class test_Functional1(Functional):
            def __init__(self, f):
                super(test_Functional1, self).__init__(f)

        f = test_Functional1(test_F1())
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))
        f.f = test_F2()
        np.testing.assert_allclose(f.evaluate(x), np.sin(x))
