from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone


cdef class Function(object):

    cpdef double evaluate_point(self, double x):
        return 0.0

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = self.evaluate_point(x[i])
        return y
