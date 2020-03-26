from BDFunction1D.Function cimport Function


cdef class InterpolateFunction(Function):
    cdef:
        double[:] __x, __y, __err
        int __n


cdef class InterpolateFunctionMesh(InterpolateFunction):
    pass
