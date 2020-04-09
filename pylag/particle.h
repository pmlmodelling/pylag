#ifndef PARTICLE_H
#define PARTICLE_H

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

            // Getters and setters
            // -------------------

            void set_x1(const double& rhs);
            double get_x1() const;

            void set_x2(const double& rhs);
            double get_x2() const;

            void set_x3(const double& rhs);
            double get_x3() const;

            void set_phi(const std::vector<double>& rhs);
            std::vector<double> get_phi() const;

            void set_omega_interfaces(const double& rhs);
            double get_omega_interfaces() const;

            void set_omega_layers(const double& rhs);
            double get_omega_layers() const;

            void set_in_domain(const bool& rhs);
            bool get_in_domain() const;

            void set_is_beached(const int& rhs);
            int get_is_beached() const;

            void set_host_horizontal_elem(const int& rhs);
            int get_host_horizontal_elem() const;

            void set_in_vertical_boundary_layer(const bool& rhs);
            bool get_in_vertical_boundary_layer() const;

            void set_k_lower_layer(const int& rhs);
            int get_k_lower_layer() const;

            void set_k_upper_layer(const int& rhs);
            int get_k_upper_layer() const;

            void set_k_layer(const int& rhs);
            int get_k_layer() const;

            // Particle properties
            // --------------------

            // Particle group ID
            int group_id;

            // Unique particle ID
            int id;

            // Status flag (0 - okay; 1 - error)
            int status;

        private:

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

            // Barycentric coordinates within the host element
            std::vector<double> phi;

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

            // The host horizontal element
            int host_horizontal_elem;

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
    };

}

#endif
