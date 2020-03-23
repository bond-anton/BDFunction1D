from BDFunction1D.Function cimport Function
from BDFunction1D.Functional cimport Functional


cdef class BinaryFunctional(Functional):
    cdef:
        Function __p


cdef class FunctionSum(BinaryFunctional):
    pass


cdef class FunctionDifference(BinaryFunctional):
    pass


cdef class FunctionMultiplication(BinaryFunctional):
    pass


cdef class FunctionDivision(BinaryFunctional):
    pass
