from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone


cdef class Function(object):

    cpdef double evaluate_point(self, double x):
        return 0.0

    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = self.evaluate_point(x[i])
        return y


cdef class ConstantFunction(Function):

    def __init__(self, double c):
        super(ConstantFunction, self).__init__()
        self.__c = c

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, double c):
        self.__c = c

    cpdef double evaluate_point(self, double x):
        return self.__c


cdef class ZeroFunction(ConstantFunction):

    def __init__(self):
        super(ZeroFunction, self).__init__(0.0)

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


cdef class LinearFunction(ConstantFunction):

    def __init__(self, double k, double c):
        super(LinearFunction, self).__init__(c)
        self.__k = k

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, double k):
        self.__k = k

    cpdef double evaluate_point(self, double x):
        return self.__k * x + self.__c
