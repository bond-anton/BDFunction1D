from BDFunction1D.Functional cimport Functional


cdef class NumericGradient(Functional):
    cdef:
        double __dx
