from libc.math cimport pow, fabs

from BDFunction1D.Function cimport Function


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


cdef class NegateFunction(Functional):

    cpdef double evaluate_point(self, double x):
        return -self.__f.evaluate_point(x)


cdef class AbsFunction(Functional):

    cpdef double evaluate_point(self, double x):
        return fabs(self.__f.evaluate_point(x))


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
