import numpy as np
from scipy.interpolate import interp1d

from BDFunction1D.Interpolation import InterpolateFunction

import unittest


class TestFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_interpolate_Function(self):
        x = np.linspace(0.0, 2 * np.pi, num=101, dtype=np.float)
        y = np.sin(x)
        f = InterpolateFunction(x, y)
        x_new = np.linspace(0.0, 2 * np.pi, num=201, dtype=np.float)
        f1 = interp1d(x, y, kind='linear')
        np.testing.assert_allclose(f.evaluate(x_new), f1(x_new), atol=1e-12)
