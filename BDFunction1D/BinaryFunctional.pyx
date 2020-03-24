from libc.math cimport pow

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

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x) + self.__p.evaluate_point(x)


cdef class FunctionDifference(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x) - self.__p.evaluate_point(x)


cdef class FunctionMultiplication(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x) * self.__p.evaluate_point(x)


cdef class FunctionDivision(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x) / self.__p.evaluate_point(x)


cdef class FunctionPower(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return pow(self.__f.evaluate_point(x), self.__p.evaluate_point(x))
