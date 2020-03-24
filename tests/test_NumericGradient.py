import numpy as np

from BDFunction1D import Function
from BDFunction1D.Interpolation import InterpolateFunction
from BDFunction1D.Differentiation import NumericGradient

import unittest


class TestFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_numeric_diff(self):

        class test_F(Function):
            def evaluate_point(self, x):
                return np.sin(x)

        f = test_F()
        df = NumericGradient(f)
        self.assertEqual(df.dx, 1e-4)
        x = np.linspace(0.0, 2 * np.pi, num=501, dtype=np.float)
        dy1 = df.evaluate(x)
        diff1 = abs(dy1 - np.cos(x))
        np.testing.assert_allclose(dy1, np.cos(x), atol=1e-4)
        df.dx = 1e-1
        self.assertEqual(df.dx, 1e-1)
        dy2 = df.evaluate(x)
        np.testing.assert_allclose(dy2, np.cos(x), atol=1e-2)
        diff2 = abs(dy2 - np.cos(x))
        self.assertTrue(np.sum(diff1) < np.sum(diff2))

        y = np.sin(x)
        f = InterpolateFunction(x, y)
        df = NumericGradient(f, 1e-4)
        np.testing.assert_allclose(df.evaluate(x), np.cos(x), atol=1e-4)
