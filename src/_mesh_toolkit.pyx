""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """

# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "mesh_toolkit.h":
    void get_barycentric_coords(double x, double y, double* x_nodes, double* y_nodes, 
                                double* phi)

# create the wrapper code, with numpy type annotations
def get_barycentric_coords_wrapper(double x, double y, 
                                   np.ndarray[double, ndim=1, mode="c"] x_nodes not None,
                                   np.ndarray[double, ndim=1, mode="c"] y_nodes not None,
                                   np.ndarray[double, ndim=1, mode="c"] phi not None):

    get_barycentric_coords(x, y,
                           <double*> np.PyArray_DATA(x_nodes),
                           <double*> np.PyArray_DATA(y_nodes),
                           <double*> np.PyArray_DATA(phi))
