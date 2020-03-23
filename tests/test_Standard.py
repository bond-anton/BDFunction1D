import math as m
import numpy as np

from BDFunction1D.Standard import Constant, Zero, Line, LineThroughPoints

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
        self.assertEqual(f.evaluate_point(x), k * x + c)
        self.assertEqual(f.evaluate_point(x), f2.evaluate_point(x))
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), f2.evaluate(x))

        f.k = 2 * f.k
        f.c = 2 * f.c
        self.assertEqual(f.k, k * 2)
        self.assertEqual(f.c, c * 2)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), 2 * k * x + 2 * c)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), 2 * k * x + 2 * c)
