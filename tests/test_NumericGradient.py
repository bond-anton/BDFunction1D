import numpy as np

from BDFunction1D import Function, InterpolateFunction, NumericGradient

import unittest


class TestFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_numeric_diff(self):

        class test_F(Function):
            def evaluate(self, x):
                return np.sin(x)

        f = test_F()
        df = NumericGradient(f)
        x = np.linspace(0.0, 2 * np.pi, num=501, dtype=np.float)
        np.testing.assert_allclose(df.evaluate(x), np.cos(x), atol=1e-4)

        y = np.sin(x)
        f = InterpolateFunction(x, y)
        df = NumericGradient(f)
        np.testing.assert_allclose(df.evaluate(x), np.cos(x), atol=1e-4)
