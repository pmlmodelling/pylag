include "constants.pxi"

from libcpp.vector cimport vector
from libc.math cimport sqrt

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t


cdef inline DTYPE_FLOAT_t float_min(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a <= b else b
cdef inline DTYPE_FLOAT_t float_max(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a >= b else b

cdef inline DTYPE_INT_t int_min(DTYPE_INT_t a, DTYPE_INT_t b): return a if a <= b else b
cdef inline DTYPE_INT_t int_max(DTYPE_INT_t a, DTYPE_INT_t b): return a if a >= b else b

cdef inline  DTYPE_FLOAT_t det_second_order(const DTYPE_FLOAT_t p1[2], const DTYPE_FLOAT_t p2[2]):
    return p1[0]*p2[1] - p1[1]*p2[0]

cdef inline DTYPE_FLOAT_t det_third_order(const DTYPE_FLOAT_t p1[3], const DTYPE_FLOAT_t p2[3], const DTYPE_FLOAT_t p3[3]):
    return p1[0]*(p2[1]*p3[2] - p3[1]*p2[2]) - p1[1]*(p2[0]*p3[2] - p3[0]*p2[2]) + p1[2]*(p2[0]*p3[1] - p3[0]*p2[1])

cdef inline DTYPE_FLOAT_t euclidian_norm(const DTYPE_FLOAT_t a[3]) except *:
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

cdef DTYPE_FLOAT_t angle_between_two_vectors(const DTYPE_FLOAT_t a[3], const DTYPE_FLOAT_t b[3]) except FLOAT_ERR

cdef void unit_vector(const DTYPE_FLOAT_t a[3], DTYPE_FLOAT_t a_unit[3]) except +

cdef inline DTYPE_FLOAT_t inner_product_two(const DTYPE_FLOAT_t a[2], const DTYPE_FLOAT_t b[2]) except *:
    return a[0]*b[0] + a[1]*b[1]

cdef inline DTYPE_FLOAT_t inner_product_three(const DTYPE_FLOAT_t a[3], const DTYPE_FLOAT_t b[3]) except *:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

cdef void vector_product(const DTYPE_FLOAT_t a[3], const DTYPE_FLOAT_t b[3], DTYPE_FLOAT_t c[3]) except +

cdef void rotate_x(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &angle, DTYPE_FLOAT_t p_new[3]) except +

cdef void rotate_y(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &angle, DTYPE_FLOAT_t p_new[3]) except +

cdef void rotate_z(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &angle, DTYPE_FLOAT_t p_new[3]) except +

cdef void rotate_axes(const DTYPE_FLOAT_t p[3],
                      const DTYPE_FLOAT_t &lon_rad,
                      const DTYPE_FLOAT_t &lat_rad,
                      DTYPE_FLOAT_t p_new[3]) except +

cdef void reverse_rotate_axes(const DTYPE_FLOAT_t p[3],
                              const DTYPE_FLOAT_t &lon_rad,
                              const DTYPE_FLOAT_t &lat_rad,
                              DTYPE_FLOAT_t p_new[3]) except +

cpdef DTYPE_FLOAT_t haversine(const DTYPE_FLOAT_t &lon1_rad,
                              const DTYPE_FLOAT_t &lat1_rad,
                              const DTYPE_FLOAT_t &lat2_rad,
                              const DTYPE_FLOAT_t &lat2_rad) except FLOAT_ERR

cdef geographic_to_cartesian_coords(const DTYPE_FLOAT_t &lon_rad,
                                    const DTYPE_FLOAT_t &lat_rad,
                                    const DTYPE_FLOAT_t &r,
                                    DTYPE_FLOAT_t cart[3])

cdef cartesian_to_geographic_coords(const DTYPE_FLOAT_t coords_cart[3], DTYPE_FLOAT_t coords_geog[2])

cdef DTYPE_INT_t get_intersection_point(const DTYPE_FLOAT_t x1[2],
                                        const DTYPE_FLOAT_t x2[2],
                                        const DTYPE_FLOAT_t x3[2],
                                        const DTYPE_FLOAT_t x4[2],
                                        DTYPE_FLOAT_t xi[2]) except INT_ERR

cdef DTYPE_INT_t get_intersection_point_in_geographic_coordinates(const DTYPE_FLOAT_t x1[2],
                                                                  const DTYPE_FLOAT_t x2[2],
                                                                  const DTYPE_FLOAT_t x3[2],
                                                                  const DTYPE_FLOAT_t x4[2],
                                                                  DTYPE_FLOAT_t xi[2]) except INT_ERR

cdef DTYPE_INT_t great_circle_arc_segments_intersect(const DTYPE_FLOAT_t x1[2],
                                                     const DTYPE_FLOAT_t x2[2],
                                                     const DTYPE_FLOAT_t x3[2],
                                                     const DTYPE_FLOAT_t x4[2]) except INT_ERR

cdef DTYPE_INT_t intersection_is_within_arc_segments(const DTYPE_FLOAT_t x1[3],
                                                     const DTYPE_FLOAT_t x2[3],
                                                     const DTYPE_FLOAT_t x3[3],
                                                     const DTYPE_FLOAT_t x4[3],
                                                     const DTYPE_FLOAT_t xi[3]) except INT_ERR


cpdef inline DTYPE_FLOAT_t sigma_to_cartesian_coords(DTYPE_FLOAT_t sigma, DTYPE_FLOAT_t h,
        DTYPE_FLOAT_t zeta):
    return zeta + sigma * (zeta - h)

cpdef inline DTYPE_FLOAT_t cartesian_to_sigma_coords(DTYPE_FLOAT_t z, DTYPE_FLOAT_t h,
        DTYPE_FLOAT_t zeta):
    return (z - zeta) / (zeta - h)

