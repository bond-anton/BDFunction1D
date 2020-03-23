from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDFunction1D.Function cimport Function
from BDFunction1D.Functional cimport Functional


cdef class BinaryFunctional(Functional):

    def __init__(self, Function f, Function p):
        super(BinaryFunctional, self).__init__(f)
        self.__p = p

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, Function p):
        self.__p = p


cdef class FunctionSum(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] + p_y[i]
        return y


cdef class FunctionDifference(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] - p_y[i]
        return y


cdef class FunctionMultiplication(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] * p_y[i]
        return y


cdef class FunctionDivision(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] / p_y[i]
        return y
