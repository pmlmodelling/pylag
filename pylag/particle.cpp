#include <vector>

#include "particle.h"

namespace particles {
    // Default constructor
    Particle::Particle()
        : group_id(-999),
          id(-999),
          status(-999),
          x1(-999.),
          x2(-999.),
          x3(-999.),
          phi(3, -999.),
          omega_interfaces(-999.),
          omega_layers(-999.),
          in_domain(false),
          is_beached(0),
          host_horizontal_elem(-999),
          k_layer(-999),
          in_vertical_boundary_layer(false),
          k_lower_layer(-999),
          k_upper_layer(-999) {
    }

    // Copy constructor
    Particle::Particle(const Particle&  rhs)
        : group_id(rhs.group_id),
          id(rhs.id),
          status(rhs.status),
          x1(rhs.x1),
          x2(rhs.x2),
          x3(rhs.x3),
          phi(rhs.phi),
          omega_interfaces(rhs.omega_interfaces),
          omega_layers(rhs.omega_layers),
          in_domain(rhs.in_domain),
          is_beached(rhs.is_beached),
          host_horizontal_elem(rhs.host_horizontal_elem),
          k_layer(rhs.k_layer),
          in_vertical_boundary_layer(rhs.in_vertical_boundary_layer),
          k_lower_layer(rhs.k_lower_layer),
          k_upper_layer(rhs.k_upper_layer) {
        }

    // Copy assignment operator
    Particle& Particle::operator=(const Particle&  rhs) {

        group_id = rhs.group_id;
        id = rhs.id;
        status = rhs.status;
        x1 = rhs.x1;
        x2 = rhs.x2;
        x3 = rhs.x3;
        phi = rhs.phi;
        omega_interfaces = rhs.omega_interfaces;
        omega_layers = rhs.omega_layers;
        in_domain = rhs.in_domain;
        is_beached = rhs.is_beached;
        host_horizontal_elem = rhs.host_horizontal_elem;
        k_layer = rhs.k_layer;
        in_vertical_boundary_layer = rhs.in_vertical_boundary_layer;
        k_lower_layer = rhs.k_lower_layer;
        k_upper_layer = rhs.k_upper_layer;

        return *this;
    }
}
