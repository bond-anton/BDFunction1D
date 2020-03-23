from libc.math cimport fabs, sqrt, pow, exp, log
from libc.math cimport sin, cos, tan, asin, acos, atan

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDFunction1D.Function cimport Function


cdef class Constant(Function):

    def __init__(self, double c):
        super(Constant, self).__init__()
        self.__c = c

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, double c):
        self.__c = c

    cpdef double evaluate_point(self, double x):
        return self.__c


cdef class Zero(Constant):

    def __init__(self):
        super(Zero, self).__init__(0.0)

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, double c):
        self.__c = 0.0

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        return clone(array('d'), x.shape[0], zero=True)


cdef class Line(Constant):

    def __init__(self, double k, double c):
        super(Line, self).__init__(c)
        self.__k = k

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, double k):
        self.__k = k

    cpdef double evaluate_point(self, double x):
        return self.__k * x + self.__c


cdef class LineThroughPoints(Line):

    def __init__(self, double x1, double y1, double x2, y2):
        cdef:
            double k, c
        k = (y2 - y1) / (x2 - x1)
        c = y1 - k * x1
        super(LineThroughPoints, self).__init__(k, c)


cdef class Abs(Function):

    cpdef double evaluate_point(self, double x):
        return fabs(x)


cdef class Pow(Function):

    def __init__(self, double exponent):
        super(Pow, self).__init__()
        self.__exp = exponent

    @property
    def exponent(self):
        return self.__exp

    @exponent.setter
    def exponent(self, double exponent):
        self.__exp = exponent

    cpdef double evaluate_point(self, double x):
        return pow(x, self.__exp)


cdef class Sqrt(Function):

    cpdef double evaluate_point(self, double x):
        return sqrt(x)


cdef class Exp(Function):

    cpdef double evaluate_point(self, double x):
        return exp(x)


cdef class Log(Function):

    cpdef double evaluate_point(self, double x):
        return log(x)


cdef class Sin(Function):

    cpdef double evaluate_point(self, double x):
        return sin(x)


cdef class Cos(Function):

    cpdef double evaluate_point(self, double x):
        return cos(x)


cdef class Tan(Function):

    cpdef double evaluate_point(self, double x):
        return tan(x)


cdef class ArcSin(Function):

    cpdef double evaluate_point(self, double x):
        return asin(x)


cdef class ArcCos(Function):

    cpdef double evaluate_point(self, double x):
        return acos(x)


cdef class ArcTan(Function):

    cpdef double evaluate_point(self, double x):
        return atan(x)
