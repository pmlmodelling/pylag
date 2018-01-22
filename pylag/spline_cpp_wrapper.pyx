cdef class PySpline:
    def __cinit__(self):
        self.c_spline = spline()

    def set_points(self, x, y):
        self.c_spline.set_points(x,y)

    def __call__(self, x):
        return self.c_spline(x)
