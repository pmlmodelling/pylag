from libcpp.vector cimport vector

cdef extern from "spline.h" namespace "tk":
    cdef cppclass spline:
        spline() except +
        void set_points(vector[double] x, vector[double] y)
        double operator()(double)

cdef class PySpline:
    cdef spline c_spline

    def __cinit__(self):
        self.c_spline = spline()

    def set_points(self, x, y):
        self.c_spline.set_points(x,y)

    def __call__(self, x):
        return self.c_spline(x)

