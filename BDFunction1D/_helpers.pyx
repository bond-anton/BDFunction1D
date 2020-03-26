import numpy as np

from libc.math cimport fabs, ceil, round, sqrt
from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDMesh.Mesh1DUniform cimport Mesh1DUniform


@boundscheck(False)
@wraparound(False)
cdef double trapz_1d(double[:] y, double[:] x):
    cdef:
        int nx = x.shape[0], ny = y.shape[0], i
        double res = 0.0
    for i in range(nx - 1):
        res += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2
    return res


@boundscheck(False)
@wraparound(False)
cdef double mean_square(double[:] x):
    cdef:
        int n = x.shape[0], i
        double res = 0.0
    for i in range(n):
        res += x[i] * x[i]
    return res / n


cdef double mean_square_root(double[:] x):
    return sqrt(mean_square(x))


@boundscheck(False)
@wraparound(False)
cdef double[:] interp_1d(double[:] x_new, double[:] x, double[:] y):
    cdef:
        int n = x_new.shape[0], m = x.shape[0]
        int i, j = 1
        array[double] y_new, template = array('d')
    y_new = clone(template, n, zero=False)
    for i in range(n):
        while x_new[i] > x[j] and j < m - 1:
            j += 1
        y_new[i] = y[j-1] + (x_new[i] - x[j-1]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
    return y_new


@boundscheck(False)
@wraparound(False)
cdef double[:] gradient1d(double[:] y, double[:] x):
    cdef:
        int i, n = x.shape[0]
        double a, b, c, dx1, dx2
        array[double] result, template = array('d')
    result = clone(template, n, zero=False)
    dx1 = x[1] - x[0]
    dx2 = x[2] - x[1]
    a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
    b = (dx1 + dx2) / (dx1 * dx2)
    c = - dx1 / (dx2 * (dx1 + dx2))
    result[0] = a * y[0] + b * y[1] + c * y[2]
    dx1 = x[n - 2] - x[n - 3]
    dx2 = x[n - 1] - x[n - 2]
    a = dx2 / (dx1 * (dx1 + dx2))
    b = - (dx2 + dx1) / (dx1 * dx2)
    c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
    result[n - 1] = a * y[n - 3] + b * y[n - 2] + c * y[n - 1]
    for i in range(1, n - 1):
        result[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return result

@boundscheck(False)
@wraparound(False)
cdef double[:] gradient1d_error(double[:] err, double[:] x):
    cdef:
        int i, n = x.shape[0]
        double a, b, c, dx1, dx2
        array[double] result, template = array('d')
    result = clone(template, n, zero=False)
    dx1 = x[1] - x[0]
    dx2 = x[2] - x[1]
    a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
    b = (dx1 + dx2) / (dx1 * dx2)
    c = - dx1 / (dx2 * (dx1 + dx2))
    result[0] = sqrt((a * err[0])**2 + (b * err[1])**2 + (c * err[2])**2)
    dx1 = x[n - 2] - x[n - 3]
    dx2 = x[n - 1] - x[n - 2]
    a = dx2 / (dx1 * (dx1 + dx2))
    b = - (dx2 + dx1) / (dx1 * dx2)
    c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
    result[n - 1] = sqrt((a * err[n - 3])**2 + (b * err[n - 2])**2 + (c * err[n - 1])**2)
    for i in range(1, n - 1):
        result[i] = sqrt(err[i + 1]**2 + err[i - 1]**2) / fabs(x[i + 1] - x[i - 1])
    return result
