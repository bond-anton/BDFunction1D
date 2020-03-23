cdef class Function(object):
    cpdef double evaluate_point(self, double x)
    cpdef double[:] evaluate(self, double[:] x)


cdef class ConstantFunction(Function):
    cdef:
        double __c


cdef class ZeroFunction(ConstantFunction):
    pass


cdef class LinearFunction(ConstantFunction):
    cdef:
        double __k
