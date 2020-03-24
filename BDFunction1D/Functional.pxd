from BDFunction1D.Function cimport Function


cdef class Functional(Function):
    cdef:
        Function __f


cdef class NegateFunction(Functional):
    pass


cdef class AbsFunction(Functional):
    pass


cdef class PowFunction(Functional):
    cdef:
        double __exp


cdef class ScaledFunction(Functional):
    cdef:
        double __scale
