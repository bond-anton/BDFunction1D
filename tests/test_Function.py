import numpy as np

from BDFunction1D import Function

import unittest


class TestFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_Function(self):
        f = Function()
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), x)

    def test_new_Function(self):
        class test_F(Function):
            def evaluate(self, x):
                return np.sqrt(x)
        f = test_F()
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))
