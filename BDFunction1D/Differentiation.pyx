from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDFunction1D._helpers cimport gradient1d
from BDFunction1D.Function cimport Function
from BDFunction1D.Functional cimport Functional
from BDFunction1D.Interpolation cimport InterpolateFunction


cdef class NumericGradient(Functional):

    def __init__(self, Function f, double dx=1e-4):
        super(NumericGradient, self).__init__(f)
        self.__dx = dx

    @property
    def dx(self):
        return self.__dx

    @dx.setter
    def dx(self, double dx):
        self.__dx = dx

    @boundscheck(False)
    @wraparound(False)
    cpdef double evaluate_point(self, double x):
        cdef:
            int j = 1
            double[:] dy
        if not isinstance(self.__f, InterpolateFunction):
            return (self.__f.evaluate_point(x + self.__dx) - self.__f.evaluate_point(x - self.__dx)) / (2 * self.__dx)
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
        y = clone(array('d'), n, zero=False)
        if not isinstance(self.__f, InterpolateFunction):
            for i in range(n):
                y[i] = self.evaluate_point(x[i])
        else:
            dy = gradient1d(self.__f.y, self.__f.x)
            for i in range(n):
                if x[i] < last_x:  # check if x is not sorted
                    j = 1
                while x[i] > self.__f.x[j] and j < self.__f.n - 1:
                    j += 1
                y[i] = dy[j-1] + (x[i] - self.__f.x[j-1]) * (dy[j] - dy[j-1]) / (self.__f.x[j] - self.__f.x[j-1])
                last_x = x[i]
        return y
