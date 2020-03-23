from BDFunction1D.Function cimport Function


cdef class Constant(Function):
    cdef:
        double __c


cdef class Zero(Constant):
    pass


cdef class Line(Constant):
    cdef:
        double __k

    cpdef through_points(self, double x1, double y1, double x2, double y2)


cdef class LineThroughPoints(Line):
    pass


cdef class Abs(Function):
    pass


cdef class Pow(Function):
    cdef:
        double __exp


cdef class Sqrt(Function):
    pass


cdef class Exp(Function):
    pass


cdef class Log(Function):
    pass


cdef class Sin(Function):
    pass


cdef class Cos(Function):
    pass


cdef class Tan(Function):
    pass


cdef class ArcSin(Function):
    pass


cdef class ArcCos(Function):
    pass


cdef class ArcTan(Function):
    pass
