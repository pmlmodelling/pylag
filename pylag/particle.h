#ifndef PARTICLE_H
#define PARTICLE_H

#include <string>
#include <unordered_map>
#include <vector>

namespace particles {

    class Particle {
        /*
         * A Particle.
         *
         * NB For now, all data members are public. Previously, Particles were implemented as structs in
         * Cython with public access to data members exploited throughout the code. This will be fixed
         * in later commits.
        */

        public:
            // Default constructor
            Particle();

            // Copy constructor
            Particle(const Particle& rhs);

            // Copy assignment operator
            Particle& operator=(const Particle& rhs);

            void clear_phis();

            void clear_host_horizontal_elems();

            void clear_parameters();

            void clear_state_variables();

            void clear_diagnostic_variables();

            void clear_boolean_flags();

            // Getters and setters
            // -------------------

            void set_group_id(const int& rhs);
            int get_group_id() const;

            void set_id(const int& rhs);
            int get_id() const;

            void set_status(const int& rhs);
            int get_status() const;

            void set_x1(const double& rhs);
            double get_x1() const;

            void set_x2(const double& rhs);
            double get_x2() const;

            void set_x3(const double& rhs);
            double get_x3() const;

            void set_phi(const std::string& grid, const std::vector<double>& rhs);
            const std::vector<double>& get_phi(const std::string& grid) const;
            void get_all_phis(std::vector<std::string>& rhs_grids, std::vector<std::vector<double>>& rhs_phis) const;

            void set_omega_interfaces(const double& rhs);
            double get_omega_interfaces() const;

            void set_omega_layers(const double& rhs);
            double get_omega_layers() const;

            void set_in_domain(const bool& rhs);
            bool get_in_domain() const;

            void set_is_beached(const int& rhs);
            int get_is_beached() const;

            void set_host_horizontal_elem(const std::string& grid, const int& host);
            int get_host_horizontal_elem(const std::string& grid) const;

            void set_all_host_horizontal_elems(const std::vector<std::string>& grids, const std::vector<int>& hosts);
            void get_all_host_horizontal_elems(std::vector<std::string>& grids, std::vector<int>& hosts) const;

            void set_in_vertical_boundary_layer(const bool& rhs);
            bool get_in_vertical_boundary_layer() const;

            void set_k_lower_layer(const int& rhs);
            int get_k_lower_layer() const;

            void set_k_upper_layer(const int& rhs);
            int get_k_upper_layer() const;

            void set_k_layer(const int& rhs);
            int get_k_layer() const;

            void set_age(const float& rhs);
            float get_age() const;

            void set_is_alive(const bool& rhs);
            bool get_is_alive() const;

           // Generic getters and setters below here, with values stores in map objects
           // -------------------------------------------------------------------------

            void set_parameter(const std::string& name, const float& value);
            float get_parameter(const std::string& name) const;

            void get_all_parameters(std::vector<std::string>& names, std::vector<float>& values) const;

            void set_state_variable(const std::string& name, const float& value);
            float get_state_variable(const std::string& name) const;

            void get_all_state_variables(std::vector<std::string>& names, std::vector<float>& values) const;

            void set_diagnostic_variable(const std::string& name, const float& value);
            float get_diagnostic_variable(const std::string& name) const;

            void get_all_diagnostic_variables(std::vector<std::string>& names, std::vector<float>& values) const;

            void set_boolean_flag(const std::string& name, const bool& value);
            float get_boolean_flag(const std::string& name) const;

            void get_all_boolean_flags(std::vector<std::string>& names, std::vector<float>& values) const;

        private:

            // Particle properties
            // --------------------

            // Particle group ID
            int group_id;

            // Unique particle ID
            int id;

            // Status flag (0 - okay; 1 - error)
            int status;

            // Global coordinates
            // ------------------

            // Particle x1-position
            double x1;

            // Particle x2-position
            double x2;

            // Particle x3-position
            double x3;

            // Local coordinates
            // -----------------

            // Barycentric coordinates within host elements. Format is: <grid_name, host>
            std::unordered_map<std::string, std::vector<double>> phis;

            // Vertical interpolation coefficient for variables defined at the interfaces
            // between k-levels
            double omega_interfaces;

            // Vertical interpolation coefficient for variables defined at the mid-point
            // of k-layers
            double omega_layers;

            // Indices describing the particle's position within a given grid
            // --------------------------------------------------------------

            // Flag identifying whether or not the particle resides within the model domain.
            bool in_domain;

            // Flag identifying whether or not a particle is beached
            int is_beached;

            // Host horizontal element. Format is: <grid_name, host>.
            std::unordered_map<std::string, int> host_elements;

            // The host k layer
            int k_layer;

            // Flag for whether the particle is in the top or bottom boundary layers
            bool in_vertical_boundary_layer;

            // Index of the k-layer lying immediately below the particle's current
            // position. Only set if the particle is not in the top or bottom boundary
            // layers
            int k_lower_layer;

            // Index of the k-layer lying immediately above the particle's current
            // position. Only set if the particle is not in the top or bottom boundary
            // layers
            int k_upper_layer;

            // Intrinsic particle properties and parameters
            // --------------------------------------------

            // Particle age in seconds
            float age;

            // Particle is living?
            bool is_alive;

            // Particle parameters. Format is: <name, value>.
            std::unordered_map<std::string, float> parameters;

            // Particle variables. Format is: <name, value>.
            std::unordered_map<std::string, float> state_variables;

            // Particle diagnostic variables. Format is: <name, value>.
            std::unordered_map<std::string, float> diagnostic_variables;

            // Particle boolean flags. Format is: <name, value>.
            std::unordered_map<std::string, float> boolean_flags;

    };

}

#endif
