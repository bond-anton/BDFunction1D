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
        self.assertEqual(f.error_point(x), 0.0)
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.zeros_like(x))
        np.testing.assert_allclose(f.error(x), np.zeros_like(x))

    def test_new_Function(self):

        class TestF(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

            def error_point(self, x):
                return 2.0

        f = TestF()
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))
        np.testing.assert_allclose(f.error(x), np.ones_like(x) * 2)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x))
        self.assertEqual(f.error_point(x), 2.0)

    def test_neg(self):

        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        f1 = TestF1()
        f = -f1

        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), -np.asarray(f1.evaluate(x)))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), -f1.evaluate_point(x))

    def test_abs(self):

        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sin(x)

        f = abs(TestF1())

        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), abs(np.sin(x)))
        x = -m.pi / 3
        self.assertEqual(f.evaluate_point(x), abs(m.sin(x)))

    def test_sum(self):

        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        class TestF2(Function):
            def evaluate_point(self, x):
                return m.sin(x)

        f1 = TestF1()
        f2 = TestF2()

        f = f1 + f2
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) + np.sin(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) + m.sin(x))

        f = -f1 + f2
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), -np.sqrt(x) + np.sin(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), -m.sqrt(x) + m.sin(x))

        f = f1 + m.pi
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) + m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) + m.pi)

        f = m.pi + f1
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) + m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) + m.pi)

        f = 2 + f1 + f2 + f1 + m.pi
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), 2 * np.sqrt(x) + np.sin(x) + 2 + m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), 2 * m.sqrt(x) + m.sin(x) + 2 + m.pi)

    def test_sub(self):

        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        class TestF2(Function):
            def evaluate_point(self, x):
                return m.sin(x)

        f1 = TestF1()
        f2 = TestF2()

        f = f1 - f2
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) - np.sin(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) - m.sin(x))

        f = f2 - f1
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sin(x) - np.sqrt(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sin(x) - m.sqrt(x))

        f = f1 - m.pi
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) - m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) - m.pi)

        f = m.pi - f1
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), -np.sqrt(x) + m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), -m.sqrt(x) + m.pi)

        f = 2 - f1 + f2 - f1 - m.pi
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), -2 * np.sqrt(x) + np.sin(x) + 2 - m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), -2 * m.sqrt(x) + m.sin(x) + 2 - m.pi)

    def test_mul(self):
        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        class TestF2(Function):
            def evaluate_point(self, x):
                return m.sin(x)

        f1 = TestF1()
        f2 = TestF2()

        f = f1 * f2
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) * np.sin(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) * m.sin(x))

        f = f1 * m.pi
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) * m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) * m.pi)

        f = m.pi * f1
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) * m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) * m.pi)

        f = (2 - f1) * (f2 - f1) - m.pi
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), (2 - np.sqrt(x)) * (np.sin(x) - np.sqrt(x)) - m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), (2 - m.sqrt(x)) * (m.sin(x) - m.sqrt(x)) - m.pi)

    def test_div(self):

        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        class TestF2(Function):
            def evaluate_point(self, x):
                return m.sin(x)

        class TestF3(Function):
            def evaluate_point(self, x):
                return x + 5

        f1 = TestF1()
        f2 = TestF2()
        f3 = TestF3()

        f = f1 / f3
        x = np.linspace(1, m.pi * 0.9, num=100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) / (x + 5))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) / (x + 5))

        f = f1 / m.pi
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) / m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) / m.pi)

        f = m.pi / f1
        x = np.arange(1, 100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), m.pi / np.sqrt(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.pi / m.sqrt(x))

        f = (2 - f1) / (f2 - f1) - m.pi
        x = np.arange(1, 100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), (2 - np.sqrt(x)) / (np.sin(x) - np.sqrt(x)) - m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), (2 - m.sqrt(x)) / (m.sin(x) - m.sqrt(x)) - m.pi)

    def test_pow(self):

        class TestF1(Function):
            def evaluate_point(self, x):
                return m.sqrt(x)

        class TestF2(Function):
            def evaluate_point(self, x):
                return m.sin(x)

        f1 = TestF1()
        f2 = TestF2()

        f = f1 ** f2
        x = np.linspace(1, m.pi * 0.9, num=100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) ** np.sin(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) ** m.sin(x))

        f = f1 ** m.pi
        x = np.linspace(1, m.pi * 0.9, num=100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x) ** m.pi)
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.sqrt(x) ** m.pi)

        f = m.pi ** f1
        x = np.linspace(1, m.pi * 0.9, num=100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), m.pi ** np.sqrt(x))
        x = m.pi
        self.assertEqual(f.evaluate_point(x), m.pi ** m.sqrt(x))
