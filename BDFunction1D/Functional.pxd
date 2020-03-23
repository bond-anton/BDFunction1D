from BDFunction1D.Function cimport Function


cdef class Functional(Function):
    cdef:
        Function __f
        Function __result


cdef class PowFunction(Functional):
    cdef:
        double __pow


cdef class ScaledFunction(Functional):
    cdef:
        double __scale


cdef class NumericGradient(Functional):
    pass
