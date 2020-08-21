include "constants.pxi"

from libcpp.vector cimport vector
from libc.math cimport sqrt


from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class Intersection:
    cdef DTYPE_FLOAT_t x1, y1, x2, y2, xi, yi

cdef inline DTYPE_FLOAT_t float_min(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a <= b else b
cdef inline DTYPE_FLOAT_t float_max(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a >= b else b

cdef inline DTYPE_INT_t int_min(DTYPE_INT_t a, DTYPE_INT_t b): return a if a <= b else b
cdef inline DTYPE_INT_t int_max(DTYPE_INT_t a, DTYPE_INT_t b): return a if a >= b else b

cpdef inline  DTYPE_FLOAT_t det_second_order(const vector[DTYPE_FLOAT_t] &p1,
                                             const vector[DTYPE_FLOAT_t] &p2):
    return p1[0]*p2[1] - p1[1]*p2[0]

cpdef inline DTYPE_FLOAT_t det_third_order(const vector[DTYPE_FLOAT_t] &p1,
                                          const vector[DTYPE_FLOAT_t] &p2,
                                          const vector[DTYPE_FLOAT_t] &p3):
    return p1[0]*(p2[1]*p3[2] - p3[1]*p2[2]) - p1[1]*(p2[0]*p3[2] - p3[0]*p2[2]) + p1[2]*(p2[0]*p3[1] - p3[0]*p2[1])

cpdef inline DTYPE_FLOAT_t euclidian_norm(const vector[DTYPE_FLOAT_t] &a) except *:
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

cpdef DTYPE_FLOAT_t angle_between_two_vectors(const vector[DTYPE_FLOAT_t] &a,
                                              const vector[DTYPE_FLOAT_t] &b) except FLOAT_ERR

cpdef vector[DTYPE_FLOAT_t] unit_vector(const vector[DTYPE_FLOAT_t] &a) except *

cpdef inline DTYPE_FLOAT_t inner_product_two(const vector[DTYPE_FLOAT_t] &a,
                                         const vector[DTYPE_FLOAT_t] &b) except *:
    return a[0]*b[0] + a[1]*b[1]

cpdef inline DTYPE_FLOAT_t inner_product_three(const vector[DTYPE_FLOAT_t] &a,
                                         const vector[DTYPE_FLOAT_t] &b) except *:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

cpdef vector[DTYPE_FLOAT_t] vector_product(const vector[DTYPE_FLOAT_t] &a,
                                           const vector[DTYPE_FLOAT_t] &b) except *

cpdef vector[DTYPE_FLOAT_t] rotate_x(const vector[DTYPE_FLOAT_t] &p, const DTYPE_FLOAT_t &angle)

cpdef vector[DTYPE_FLOAT_t] rotate_y(const vector[DTYPE_FLOAT_t] &p, const DTYPE_FLOAT_t &angle)

cpdef vector[DTYPE_FLOAT_t] rotate_z(const vector[DTYPE_FLOAT_t] &p, const DTYPE_FLOAT_t &angle)

cpdef vector[DTYPE_FLOAT_t] rotate_axes(const vector[DTYPE_FLOAT_t] &p,
                                        const DTYPE_FLOAT_t &lon_rad,
                                        const DTYPE_FLOAT_t &lat_rad)

cpdef DTYPE_FLOAT_t haversine(const DTYPE_FLOAT_t &lon1_rad,
                              const DTYPE_FLOAT_t &lat1_rad,
                              const DTYPE_FLOAT_t &lat2_rad,
                              const DTYPE_FLOAT_t &lat2_rad) except FLOAT_ERR

cpdef vector[DTYPE_FLOAT_t] geographic_to_cartesian_coords(const DTYPE_FLOAT_t &lon_rad,
                                                           const DTYPE_FLOAT_t &lat_rad,
                                                           const DTYPE_FLOAT_t &r)

cdef DTYPE_INT_t get_intersection_point(const vector[DTYPE_FLOAT_t] &x1,
                                        const vector[DTYPE_FLOAT_t] &x2,
                                        const vector[DTYPE_FLOAT_t] &x3,
                                        const vector[DTYPE_FLOAT_t] &x4,
                                        vector[DTYPE_FLOAT_t] &xi) except INT_ERR

cpdef DTYPE_INT_t great_circle_arc_segments_intersect(const vector[DTYPE_FLOAT_t] &x1,
                                                      const vector[DTYPE_FLOAT_t] &x2,
                                                      const vector[DTYPE_FLOAT_t] &x3,
                                                      const vector[DTYPE_FLOAT_t] &x4) except INT_ERR

cpdef DTYPE_INT_t intersection_is_within_arc_segments(const vector[DTYPE_FLOAT_t] &x1,
                                                      const vector[DTYPE_FLOAT_t] &x2,
                                                      const vector[DTYPE_FLOAT_t] &x3,
                                                      const vector[DTYPE_FLOAT_t] &x4,
                                                      const vector[DTYPE_FLOAT_t] &xi) except INT_ERR



cpdef inline DTYPE_FLOAT_t sigma_to_cartesian_coords(DTYPE_FLOAT_t sigma, DTYPE_FLOAT_t h,
        DTYPE_FLOAT_t zeta):
    return zeta + sigma * (zeta - h)

cpdef inline DTYPE_FLOAT_t cartesian_to_sigma_coords(DTYPE_FLOAT_t z, DTYPE_FLOAT_t h,
        DTYPE_FLOAT_t zeta):
    return (z - zeta) / (zeta - h)
