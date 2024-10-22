from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp cimport bool

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t


cdef extern from "particle.cpp":
    pass

cdef extern from "particle.h" namespace "particles":
    cdef cppclass Particle:
        Particle() except +
        Particle(const Particle&) except +
        Particle& operator=(const Particle&) except +

        void clear_phis() except +

        void clear_host_horizontal_elems() except +

        void clear_parameters() except +

        void clear_state_variables() except +

        void clear_diagnostic_variables() except +

        void clear_boolean_flags() except +

        void set_group_id(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_group_id() except +

        void set_id(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_id() except +

        void set_status(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_status() except +

        void set_x1(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_x1() except +

        void set_x2(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_x2() except +

        void set_x3(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_x3() except +

        void set_phi(const string&, const vector[DTYPE_FLOAT_t]&) except +
        const vector[DTYPE_FLOAT_t]& get_phi(const string&) except +
        void get_all_phis(vector[string]&, vector[vector[DTYPE_FLOAT_t]]&) except +

        void set_omega_interfaces(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_omega_interfaces() except +

        void set_omega_layers(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_omega_layers() except +

        void set_in_domain(const bint&) except +
        bint get_in_domain() except +

        void set_is_beached(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_is_beached() except +

        void set_host_horizontal_elem(const string&, const DTYPE_INT_t&) except +
        DTYPE_INT_t get_host_horizontal_elem(const string&) except +

        void set_all_host_horizontal_elems(const vector[string]&, const vector[int]&) except +
        void get_all_host_horizontal_elems(vector[string]&, vector[int]&) except +

        void set_k_layer(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_k_layer() except +

        void set_in_surface_boundary_layer(const bint&) except +
        bint get_in_surface_boundary_layer() except +

        void set_in_bottom_boundary_layer(const bint&) except +
        bint get_in_bottom_boundary_layer() except +

        void set_k_lower_layer(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_k_lower_layer() except +

        void set_k_upper_layer(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_k_upper_layer() except +

        void set_restore_to_fixed_depth(const bint&) except +
        bint get_restore_to_fixed_depth() except +

        void set_fixed_depth(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_fixed_depth() except +

        void set_restore_to_fixed_height(const bint&) except +
        bint get_restore_to_fixed_height() except +

        void set_fixed_height(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_fixed_height() except +

        void set_age(const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_age() except +

        void set_is_alive(const bint&) except +
        bint get_is_alive() except +

        void set_land_boundary_encounters(const DTYPE_INT_t&) except +
        DTYPE_INT_t get_land_boundary_encounters() except +
        void register_land_boundary_encounter() except +

        void set_parameter(const string&, const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_parameter(const string&) except +

        void get_all_parameters(vector[string]&, vector[float]&) except +

        void set_state_variable(const string&, const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_state_variable(const string&) except +

        void get_all_state_variables(vector[string]&, vector[float]&) except +

        void set_diagnostic_variable(const string&, const DTYPE_FLOAT_t&) except +
        DTYPE_FLOAT_t get_diagnostic_variable(const string&) except +

        void get_all_diagnostic_variables(vector[string]&, vector[float]&) except +

        void set_boolean_flag(const string&, const bool&) except +
        bint get_boolean_flag(const string&) except +

        void get_all_boolean_flags(vector[string]&, vector[bool]&) except +

