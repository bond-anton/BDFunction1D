import math as m
import numpy as np

from BDFunction1D.Standard import Constant, Zero, Line, LineThroughPoints, Abs
from BDFunction1D.Standard import Pow, Sqrt, Exp, Log
from BDFunction1D.Standard import Sin, Cos, Tan, ArcSin, ArcCos, ArcTan

import unittest


class TestStandard(unittest.TestCase):

    def setUp(self):
        pass

    def test_Constant(self):
        f = Constant(m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.pi)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.ones_like(x) * m.pi)

        f.c = 2 * m.pi
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.pi * 2)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.ones_like(x) * m.pi * 2)

    def test_Zero(self):
        f = Zero()
        x = m.pi
        self.assertEqual(f.evaluate_point(x), 0.0)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.zeros_like(x))

        f.c = 2 * m.pi
        x = m.pi
        self.assertEqual(f.evaluate_point(x), 0.0)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.zeros_like(x))

    def test_Line(self):
        k = 2.0
        c = 3.0
        f = Line(k, c)
        self.assertEqual(f.k, k)
        self.assertEqual(f.c, c)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), k * x + c)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), k * x + c)

        f.k = 2 * f.k
        f.c = 2 * f.c
        self.assertEqual(f.k, k * 2)
        self.assertEqual(f.c, c * 2)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), 2 * k * x + 2 * c)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), 2 * k * x + 2 * c)

        x1 = 2.0
        y1 = 3.0
        x2 = 5.0
        y2 = 7.0
        k = (y2 - y1) / (x2 - x1)
        c = y1 - (y2 - y1) / (x2 - x1) * x1
        f.through_points(x1, y1, x2, y2)
        self.assertEqual(f.k, k)
        self.assertEqual(f.c, c)
        self.assertEqual(f.y_intercept, c)
        self.assertEqual(f.x_intercept, -c / k)

        f.y_intercept = 3.0
        self.assertEqual(f.c, 3.0)
        f.x_intercept = 2.0
        self.assertEqual(f.k, -1.5)

    def test_LineThroughPoints(self):
        x1 = 2.0
        y1 = 3.0
        x2 = 5.0
        y2 = 7.0
        k = (y2 - y1) / (x2 - x1)
        c = y1 - (y2 - y1) / (x2 - x1) * x1
        f = LineThroughPoints(x1, y1, x2, y2)
        f2 = Line(k, c)
        self.assertEqual(f.k, k)
        self.assertEqual(f.c, c)
        x = m.pi
        self.assertAlmostEqual(f.evaluate_point(x), k * x + c, places=14)
        self.assertEqual(f.evaluate_point(x), f2.evaluate_point(x))
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), f2.evaluate(x))

        f.k = 2 * f.k
        f.c = 2 * f.c
        self.assertEqual(f.k, k * 2)
        self.assertEqual(f.c, c * 2)
        x = m.pi
        self.assertAlmostEqual(f.evaluate_point(x), 2 * k * x + 2 * c, places=14)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), 2 * k * x + 2 * c)

    def test_Abs(self):
        f = Abs()
        x = m.pi
        self.assertEqual(f.evaluate_point(x), x)
        self.assertEqual(f.evaluate_point(-x), x)
        x = np.arange(100, dtype=np.float) - 50.0
        np.testing.assert_allclose(f.evaluate(x), np.abs(x))

    def test_Pow(self):
        exponent = 2.5
        f = Pow(exponent)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), x ** exponent)
        x = np.arange(100, dtype=np.float) - 50.0
        np.testing.assert_allclose(f.evaluate(x), x ** exponent)
        exponent = -2.33
        f.exponent = exponent
        x = m.pi
        self.assertEqual(f.evaluate_point(x), x ** exponent)
        x = np.arange(100, dtype=np.float) - 50.0
        np.testing.assert_allclose(f.evaluate(x), x ** exponent)

    def test_Sqrt(self):
        f = Sqrt()
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x))
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))

    def test_Exp(self):
        f = Exp()
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.exp(x))
        self.assertEqual(f.evaluate_point(-x), m.exp(-x))
        x = np.arange(100, dtype=np.float) - 50
        np.testing.assert_allclose(f.evaluate(x), np.exp(x))

    def test_Log(self):
        f = Log()
        f2 = Exp()
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.log(x))
        self.assertAlmostEqual(f2.evaluate_point(f.evaluate_point(x)), x, places=14)
        x = np.arange(100, dtype=np.float) + 1.0
        np.testing.assert_allclose(f.evaluate(x), np.log(x))
        np.testing.assert_allclose(f2.evaluate(f.evaluate(x)), x)

    def test_Sin(self):
        f = Sin()
        f2 = ArcSin()
        x = m.pi / 7
        self.assertEqual(f.evaluate_point(x), m.sin(x))
        self.assertEqual(f2.evaluate_point(x), m.asin(x))
        self.assertAlmostEqual(f2.evaluate_point(f.evaluate_point(x)), x, places=14)
        x = np.linspace(-np.pi / 2, np.pi / 2, num=100, endpoint=True, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sin(x))
        np.testing.assert_allclose(f2.evaluate(f.evaluate(x)), x)

    def test_Cos(self):
        f = Cos()
        f2 = ArcCos()
        x = m.pi / 7
        self.assertEqual(f.evaluate_point(x), m.cos(x))
        self.assertEqual(f2.evaluate_point(x), m.acos(x))
        self.assertAlmostEqual(f2.evaluate_point(f.evaluate_point(x)), x, places=14)
        x = np.linspace(0, np.pi, num=100, endpoint=True, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.cos(x))
        np.testing.assert_allclose(f2.evaluate(f.evaluate(x)), x)

    def test_Tan(self):
        f = Tan()
        f2 = ArcTan()
        x = m.pi / 7
        self.assertEqual(f.evaluate_point(x), m.tan(x))
        self.assertEqual(f2.evaluate_point(x), m.atan(x))
        self.assertAlmostEqual(f2.evaluate_point(f.evaluate_point(x)), x, places=14)
        x = np.linspace(-np.pi / 2, np.pi / 2, num=100, endpoint=True, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.tan(x))
        np.testing.assert_allclose(f2.evaluate(f.evaluate(x)), x)
