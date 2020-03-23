from BDFunction1D.Function cimport Function


cdef class InterpolateFunction(Function):
    cdef:
        double[:] __x, __y
        int __n


cdef class InterpolateFunctionMesh(InterpolateFunction):
    pass
