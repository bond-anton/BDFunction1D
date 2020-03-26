from libc.math cimport fabs, sqrt
from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDMesh.Mesh1D cimport Mesh1D
from BDMesh.TreeMesh1D cimport TreeMesh1D

from BDFunction1D.Function cimport Function


cdef class InterpolateFunction(Function):

    def __init__(self, double[:] x, double[:] y, double[:] err=None):
        super(InterpolateFunction, self).__init__()
        self.__x = x
        self.__y = y
        self.__n = x.shape[0]
        if err is None:
            self.__err = clone(array('d'), self.__n, zero=False)
        else:
            self.__err = err

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def err(self):
        return self.__err

    @property
    def n(self):
        return self.__n

    @boundscheck(False)
    @wraparound(False)
    cpdef double evaluate_point(self, double x):
        cdef:
            int j = 1
        while x > self.__x[j] and j < self.__n - 1:
            j += 1
        return self.__y[j-1] + (x - self.__x[j-1]) * \
               (self.__y[j] - self.__y[j-1]) / (self.__x[j] - self.__x[j-1])

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, j = 1, n = x.shape[0]
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            while x[i] > self.__x[j] and j < self.__n - 1:
                j += 1
            y[i] = self.__y[j-1] + (x[i] - self.__x[j-1]) * \
                   (self.__y[j] - self.__y[j-1]) / (self.__x[j] - self.__x[j-1])
        return y

    @boundscheck(False)
    @wraparound(False)
    cpdef double error_point(self, double x):
        cdef:
            int j = 1
        while x > self.__x[j] and j < self.__n - 1:
            j += 1
        return sqrt(self.__err[j-1]**2 + (x - self.__x[j-1])**2
                    * (self.__err[j]**2 + self.__err[j-1]**2))\
               / fabs(self.__x[j] - self.__x[j-1])

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] error(self, double[:] x):
        cdef:
            int i, j = 1, n = x.shape[0]
            array[double] err = clone(array('d'), n, zero=False)
        for i in range(n):
            while x[i] > self.__x[j] and j < self.__n - 1:
                j += 1
            err[i] = sqrt(self.__err[j-1]**2 + (x[i] - self.__x[j-1])**2
                          * (self.__err[j]**2 + self.__err[j-1]**2))\
                     / fabs(self.__x[j] - self.__x[j-1])
        return err


cdef class InterpolateFunctionMesh(InterpolateFunction):

    def __init__(self, mesh):
        if isinstance(mesh, TreeMesh1D):
            flat_mesh = mesh.flatten()
            x = flat_mesh.physical_nodes
            y = flat_mesh.solution
            err = flat_mesh.residual
        elif isinstance(mesh, Mesh1D):
            x = mesh.physical_nodes
            y = mesh.solution
            err = mesh.residual
        super(InterpolateFunctionMesh, self).__init__(x, y, err)
