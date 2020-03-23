from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDMesh.Mesh1D cimport Mesh1D
from BDMesh.TreeMesh1D cimport TreeMesh1D

from BDFunction1D.Function cimport Function


cdef class InterpolateFunction(Function):

    def __init__(self, double[:] x, double[:] y):
        super(InterpolateFunction, self).__init__()
        self.__x = x
        self.__y = y
        self.__n = x.shape[0]

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

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


cdef class InterpolateFunctionMesh(InterpolateFunction):

    def __init__(self, mesh):
        if isinstance(mesh, TreeMesh1D):
            flat_mesh = mesh.flatten()
            x = flat_mesh.physical_nodes()
            y = flat_mesh.solution
        elif isinstance(mesh, Mesh1D):
            x = mesh.physical_nodes()
            y = mesh.solution
        super(InterpolateFunctionMesh, self).__init__(x, y)
