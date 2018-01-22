from libcpp.vector cimport vector

cdef extern from "spline.h" namespace "tk":
    cdef cppclass spline:
        spline() except +
        void set_points(vector[double] x, vector[double] y)
        double operator()(double)

cdef class PySpline:
    cdef spline c_spline