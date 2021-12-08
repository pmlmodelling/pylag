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
          phis{},
          omega_interfaces(-999.),
          omega_layers(-999.),
          in_domain(false),
          is_beached(0),
          host_elements{},
          k_layer(-999),
          in_vertical_boundary_layer(false),
          k_lower_layer(-999),
          k_upper_layer(-999),
          restore_to_fixed_depth(false),
          fixed_depth(-999.),
          age(-999.),
          is_alive(false),
          parameters{},
          state_variables{},
          diagnostic_variables{},
          boolean_flags{} {
    }

    // Copy constructor
    Particle::Particle(const Particle&  rhs)
        : group_id(rhs.group_id),
          id(rhs.id),
          status(rhs.status),
          x1(rhs.x1),
          x2(rhs.x2),
          x3(rhs.x3),
          phis(rhs.phis),
          omega_interfaces(rhs.omega_interfaces),
          omega_layers(rhs.omega_layers),
          in_domain(rhs.in_domain),
          is_beached(rhs.is_beached),
          host_elements(rhs.host_elements),
          k_layer(rhs.k_layer),
          in_vertical_boundary_layer(rhs.in_vertical_boundary_layer),
          k_lower_layer(rhs.k_lower_layer),
          k_upper_layer(rhs.k_upper_layer),
          restore_to_fixed_depth(rhs.restore_to_fixed_depth),
          fixed_depth(rhs.fixed_depth),
          age(rhs.age),
          is_alive(rhs.is_alive),
          parameters(rhs.parameters),
          state_variables(rhs.state_variables),
          diagnostic_variables(rhs.diagnostic_variables),
          boolean_flags(rhs.boolean_flags) {
    }

    // Copy assignment operator
    Particle& Particle::operator=(const Particle&  rhs) {

        group_id = rhs.group_id;
        id = rhs.id;
        status = rhs.status;
        x1 = rhs.x1;
        x2 = rhs.x2;
        x3 = rhs.x3;
        phis = rhs.phis;
        omega_interfaces = rhs.omega_interfaces;
        omega_layers = rhs.omega_layers;
        in_domain = rhs.in_domain;
        is_beached = rhs.is_beached;
        host_elements = rhs.host_elements;
        k_layer = rhs.k_layer;
        in_vertical_boundary_layer = rhs.in_vertical_boundary_layer;
        k_lower_layer = rhs.k_lower_layer;
        k_upper_layer = rhs.k_upper_layer;
        restore_to_fixed_depth = rhs.restore_to_fixed_depth;
        fixed_depth = rhs.fixed_depth;
        age = rhs.age;
        is_alive = rhs.is_alive;
        parameters = rhs.parameters;
        state_variables = rhs.state_variables;
        diagnostic_variables = rhs.diagnostic_variables;
        boolean_flags = rhs.boolean_flags;

        return *this;
    }

    void Particle::clear_phis() {
        phis.clear();
    }

    void Particle::clear_host_horizontal_elems() {
        host_elements.clear();
    }

    void Particle::clear_parameters() {
        parameters.clear();
    }

    void Particle::clear_state_variables() {
        state_variables.clear();
    }

    void Particle::clear_diagnostic_variables() {
        diagnostic_variables.clear();
    }

    void Particle::clear_boolean_flags() {
        boolean_flags.clear();
    }

    // Getters and setters
    // -------------------
    void Particle::set_group_id(const int& rhs) {
        group_id = rhs;
    }

    int Particle::get_group_id() const {
        return group_id;
    }

    void Particle::set_id(const int& rhs) {
        id = rhs;
    }

    int Particle::get_id() const {
        return id;
    }

    void Particle::set_status(const int& rhs) {
        status = rhs;
    }

    int Particle::get_status() const {
        return status;
    }

    void Particle::set_x1(const double& rhs) {
        x1 = rhs;
    }

    double Particle::get_x1() const {
        return x1;
    }

    void Particle::set_x2(const double& rhs) {
        x2 = rhs;
    }

    double Particle::get_x2() const {
        return x2;
    }

    void Particle::set_x3(const double& rhs) {
        x3 = rhs;
    }

    double Particle::get_x3() const {
        return x3;
    }

    void Particle::set_phi(const std::string& grid, const std::vector<double>& rhs) {
        phis[grid] = rhs;
    }

    const std::vector<double>& Particle::get_phi(const std::string& grid) const {
        return phis.at(grid);
    }

    void Particle::get_all_phis(std::vector<std::string>& rhs_grids, std::vector<std::vector<double>>& rhs_phis) const {
        for (auto& x: phis) {
            rhs_grids.push_back(x.first);
            rhs_phis.push_back(x.second);
        }
    }

    void Particle::set_omega_interfaces(const double& rhs) {
        omega_interfaces = rhs;
    }

    double Particle::get_omega_interfaces() const {
        return omega_interfaces;
    }

    void Particle::set_omega_layers(const double& rhs) {
        omega_layers = rhs;
    }

    double Particle::get_omega_layers() const {
        return omega_layers;
    }

    void Particle::set_in_domain(const bool& rhs) {
        in_domain = rhs;
    }

    bool Particle::get_in_domain() const {
        return in_domain;
    }

    void Particle::set_is_beached(const int& rhs) {
        is_beached = rhs;
    }

    int Particle::get_is_beached() const {
        return is_beached;
    }

    void Particle::set_host_horizontal_elem(const std::string& grid, const int& host) {
        host_elements[grid] = host;
    }

    int Particle::get_host_horizontal_elem(const std::string& grid) const {
        return host_elements.at(grid);
    }

    void Particle::set_all_host_horizontal_elems(const std::vector<std::string>& grids, const std::vector<int>& hosts) {
        host_elements.clear();

        auto igrid = grids.begin();
        auto ihost = hosts.begin();
        for (; igrid != grids.end() and ihost != hosts.end(); ++igrid, ++ihost) {
            host_elements[*igrid] = *ihost;
        }
    }

    void Particle::get_all_host_horizontal_elems(std::vector<std::string>& grids, std::vector<int>& hosts) const {
        for (auto& x: host_elements) {
            grids.push_back(x.first);
            hosts.push_back(x.second);
        }
    }

    void Particle::set_k_layer(const int& rhs) {
        k_layer = rhs;
    }

    int Particle::get_k_layer() const {
        return k_layer;
    }

    void Particle::set_in_vertical_boundary_layer(const bool& rhs) {
        in_vertical_boundary_layer = rhs;
    }

    bool Particle::get_in_vertical_boundary_layer() const {
        return in_vertical_boundary_layer;
    }

    void Particle::set_k_lower_layer(const int& rhs) {
        k_lower_layer = rhs;
    }

    int Particle::get_k_lower_layer() const {
        return k_lower_layer;
    }

    void Particle::set_k_upper_layer(const int& rhs) {
        k_upper_layer = rhs;
    }

    int Particle::get_k_upper_layer() const {
        return k_upper_layer;
    }

    void Particle::set_restore_to_fixed_depth(const bool& rhs) {
        restore_to_fixed_depth = rhs;
    }

    bool Particle::get_restore_to_fixed_depth() const {
        return restore_to_fixed_depth;
    }

    void Particle::set_fixed_depth(const double& rhs) {
        fixed_depth = rhs;
    }

    double Particle::get_fixed_depth() const {
        return fixed_depth;
    }

    void Particle::set_age(const float& rhs) {
        age = rhs;
    }

    float Particle::get_age() const {
        return age;
    }

    bool Particle::get_is_alive() const {
        return is_alive;
    }

    void Particle::set_is_alive(const bool& rhs) {
        is_alive = rhs;
    }

    void Particle::set_parameter(const std::string& name, const float& value) {
        parameters[name] = value;
    }

    float Particle::get_parameter(const std::string& name) const {
        return parameters.at(name);
    }

    void Particle::get_all_parameters(std::vector<std::string>& names, std::vector<float>& values) const {
        for (auto& x: parameters) {
            names.push_back(x.first);
            values.push_back(x.second);
        }
    }

    void Particle::set_state_variable(const std::string& name, const float& value) {
        state_variables[name] = value;
    }

    float Particle::get_state_variable(const std::string& name) const {
        return state_variables.at(name);
    }

    void Particle::get_all_state_variables(std::vector<std::string>& names, std::vector<float>& values) const {
        for (auto& x: state_variables) {
            names.push_back(x.first);
            values.push_back(x.second);
        }
    }

    void Particle::set_diagnostic_variable(const std::string& name, const float& value) {
        diagnostic_variables[name] = value;
    }

    float Particle::get_diagnostic_variable(const std::string& name) const {
        return diagnostic_variables.at(name);
    }

    void Particle::get_all_diagnostic_variables(std::vector<std::string>& names, std::vector<float>& values) const {
        for (auto& x: diagnostic_variables) {
            names.push_back(x.first);
            values.push_back(x.second);
        }
    }

    void Particle::set_boolean_flag(const std::string& name, const bool& value) {
        boolean_flags[name] = value;
    }

    bool Particle::get_boolean_flag(const std::string& name) const {
        return boolean_flags.at(name);
    }

    void Particle::get_all_boolean_flags(std::vector<std::string>& names, std::vector<bool>& values) const {
        for (auto& x: boolean_flags) {
            names.push_back(x.first);
            values.push_back(x.second);
        }
    }
}
