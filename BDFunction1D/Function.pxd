cdef class Function(object):

    cpdef double evaluate_point(self, double x)
    cpdef double[:] evaluate(self, double[:] x)

    cpdef double error_point(self, double x)
    cpdef double[:] error(self, double[:] x)
