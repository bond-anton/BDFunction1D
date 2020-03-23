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

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        return self.__f.evaluate(x)

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

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = self.__scale * f_y[i]
        return y


cdef class PowFunction(Functional):

    def __init__(self, Function f, double exp):
        super(PowFunction, self).__init__(f)
        self.__exp = exp

    @property
    def exp(self):
        return self.__exp

    @exp.setter
    def exp(self, double exp):
        self.__exp = exp

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = pow(f_y[i], self.__exp)
        return y


cdef class NumericGradient(Functional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, j = 1, n = x.shape[0]
            array[double] y
            double[:] dy
        if not isinstance(self.__f, InterpolateFunction):
            return gradient1d(self.__f.evaluate(x), x)
        else:
            y = clone(array('d'), n, zero=False)
            dy = gradient1d(self.__f.y, self.__f.x)
            for i in range(n):
                while x[i] > self.__f.x[j] and j < self.__f.n - 1:
                    j += 1
                y[i] = dy[j-1] + (x[i] - self.__f.x[j-1]) * (dy[j] - dy[j-1]) / (self.__f.x[j] - self.__f.x[j-1])
            return y
