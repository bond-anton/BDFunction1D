from libc.math cimport pow

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone
from BDFunction1D._helpers cimport gradient1d

from BDFunction1D.Function cimport Function
from BDFunction1D.Interpolation cimport InterpolateFunction


cdef class Functional(Function):

    def __init__(self, Function f):
        super(Functional, self).__init__()
        self.__f = f

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x)

    @property
    def f(self):
        return self.__f

    @f.setter
    def f(self, Function f):
        self.__f = f


cdef class ScaledFunction(Functional):

    def __init__(self, Function f, double scale):
        super(ScaledFunction, self).__init__(f)
        self.__scale = scale

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, double scale):
        self.__scale = scale

    cpdef double evaluate_point(self, double x):
        return self.__scale * self.__f.evaluate_point(x)


cdef class PowFunction(Functional):

    def __init__(self, Function f, double exponent):
        super(PowFunction, self).__init__(f)
        self.__exp = exponent

    @property
    def exponent(self):
        return self.__exp

    @exponent.setter
    def exponent(self, double exponent):
        self.__exp = exponent

    cpdef double evaluate_point(self, double x):
        return pow(self.__f.evaluate_point(x), self.__exp)


cdef class NumericGradient(Functional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double evaluate_point(self, double x):
        cdef:
            double h
            int j = 1
            double[:] dy
        if not isinstance(self.__f, InterpolateFunction):
            h = x * 1e-3
            return (self.__f.evaluate_point(x + h) - self.__f.evaluate_point(x - h)) / (2 * h)
        else:
            dy = gradient1d(self.__f.y, self.__f.x)
            while x > self.__f.x[j] and j < self.__f.n - 1:
                j += 1
            return dy[j-1] + (x - self.__f.x[j-1]) * (dy[j] - dy[j-1]) / (self.__f.x[j] - self.__f.x[j-1])

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, j = 1, n = x.shape[0]
            double last_x = x[0]
            array[double] y
            double[:] dy
        if not isinstance(self.__f, InterpolateFunction):
            return gradient1d(self.__f.evaluate(x), x)
        else:
            y = clone(array('d'), n, zero=False)
            dy = gradient1d(self.__f.y, self.__f.x)
            for i in range(n):
                if x[i] < last_x:
                    j = 1
                while x[i] > self.__f.x[j] and j < self.__f.n - 1:
                    j += 1
                y[i] = dy[j-1] + (x[i] - self.__f.x[j-1]) * (dy[j] - dy[j-1]) / (self.__f.x[j] - self.__f.x[j-1])
                last_x = x[i]
            return y
