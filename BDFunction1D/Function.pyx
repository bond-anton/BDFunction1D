import numbers

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDFunction1D.Standard cimport Constant
from BDFunction1D.Functional cimport ScaledFunction, PowFunction, NegateFunction, AbsFunction
from BDFunction1D.BinaryFunctional cimport FunctionSum, FunctionDifference
from BDFunction1D.BinaryFunctional cimport FunctionMultiplication, FunctionDivision
from BDFunction1D.BinaryFunctional cimport FunctionPower


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

    cpdef double error_point(self, double x):
        return 0.0

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] error(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = self.error_point(x[i])
        return y

    def __neg__(self):
        return NegateFunction(self)

    def __abs__(self):
        return AbsFunction(self)

    def __mul__(x, y):
        if isinstance(x, Function) and isinstance(y, Function):
            return FunctionMultiplication(x, y)
        elif isinstance(x, Function) and isinstance(y, numbers.Number):
            return ScaledFunction(x, <double> y)
        elif isinstance(x, numbers.Number) and isinstance(y, Function):
            return ScaledFunction(y, <double> x)
        else:
            return NotImplemented

    def __add__(x, y):
        if isinstance(x, Function) and isinstance(y, Function):
            return FunctionSum(x, y)
        elif isinstance(x, Function) and isinstance(y, numbers.Number):
            return FunctionSum(x, Constant(<double> y))
        elif isinstance(x, numbers.Number) and isinstance(y, Function):
            return FunctionSum(y, Constant(<double> x))
        else:
            return NotImplemented

    def __sub__(x, y):
        if isinstance(x, Function) and isinstance(y, Function):
            return FunctionDifference(x, y)
        elif isinstance(x, Function) and isinstance(y, numbers.Number):
            return FunctionDifference(x, Constant(<double> y))
        elif isinstance(x, numbers.Number) and isinstance(y, Function):
            return FunctionDifference(Constant(<double> x), y)
        else:
            return NotImplemented

    def __div__(x, y):
        if isinstance(x, Function) and isinstance(y, Function):
            return FunctionDivision(x, y)
        elif isinstance(x, Function) and isinstance(y, numbers.Number):
            return ScaledFunction(x, 1.0 / <double> y)
        elif isinstance(x, numbers.Number) and isinstance(y, Function):
            return FunctionDivision(Constant(<double> x), y)
        else:
            return NotImplemented

    def __truediv__(x, y):
        if isinstance(x, Function) and isinstance(y, Function):
            return FunctionDivision(x, y)
        elif isinstance(x, Function) and isinstance(y, numbers.Number):
            return ScaledFunction(x, 1.0 / <double> y)
        elif isinstance(x, numbers.Number) and isinstance(y, Function):
            return FunctionDivision(Constant(<double> x), y)
        else:
            return NotImplemented

    def __pow__(x, power, modulo):
        if isinstance(x, Function) and isinstance(power, Function):
            return FunctionPower(x, power)
        elif isinstance(x, Function) and isinstance(power, numbers.Number):
            return PowFunction(x, <double> power)
        elif isinstance(x, numbers.Number) and isinstance(power, Function):
            return FunctionPower(Constant(<double> x), power)
        else:
            return NotImplemented
