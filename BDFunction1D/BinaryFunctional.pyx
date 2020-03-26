from libc.math cimport fabs, pow, sqrt, log, HUGE_VAL

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

    cpdef double error_point(self, double x):
        return sqrt(self.__f.error_point(x)**2 + self.__p.error_point(x)**2)


cdef class FunctionDifference(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x) - self.__p.evaluate_point(x)

    cpdef double error_point(self, double x):
        return sqrt(self.__f.error_point(x)**2 + self.__p.error_point(x)**2)


cdef class FunctionMultiplication(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x) * self.__p.evaluate_point(x)

    cpdef double error_point(self, double x):
        return sqrt((self.__p.evaluate_point(x) * self.__f.error_point(x))**2
                    + (self.__f.evaluate_point(x) * self.__p.error_point(x))**2)


cdef class FunctionDivision(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return self.__f.evaluate_point(x) / self.__p.evaluate_point(x)

    cpdef double error_point(self, double x):
        return sqrt((self.__f.error_point(x))**2 / self.__p.evaluate_point(x)
                    + (self.__f.evaluate_point(x) * self.__p.error_point(x) / (self.__p.evaluate_point(x)**2))**2)


cdef class FunctionPower(BinaryFunctional):

    cpdef double evaluate_point(self, double x):
        return pow(self.__f.evaluate_point(x), self.__p.evaluate_point(x))

    cpdef double error_point(self, double x):
        if self.__f.evaluate_point(x) != 0:
            return sqrt(
                (pow(self.__f.evaluate_point(x), self.__p.evaluate_point(x)) *
                 log(fabs(self.__f.evaluate_point(x))) * self.__p.error_point(x))**2 +
                (pow(self.__f.evaluate_point(x), self.__p.evaluate_point(x) - 1) *
                 self.__p.evaluate_point(x) * self.__f.error_point(x))**2
            )
        else:
            return HUGE_VAL
