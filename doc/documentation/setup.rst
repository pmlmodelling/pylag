.. _setup:

Setup
=====

Each simulation requires a small number of input files. The first of these is a
`run configuration file <run_configuration_file_>`_, which is parsed by PyLag
on start up, and gives values for parameters that define the type of simulation
to be performed. The second is a
`particle initial positions file <particle_initial_positions_file_>`_, which
gives the number of particles and their initial positions. Finally, if
field data is to be read in from file, these files must also be provided.

.. _run_configuration_file:

Run configuration file
----------------------

Full simulations require a run configuration file. This can be given on the
command line. A basic run configuration file may look like:

.. literalinclude:: text/pylag_example.cfg

When running PyLag from the command line the run configuration file can be 
passed in as a command line argument. For example:

    python -m pylag.main -c pylag.cfg


.. _particle_initial_positions_file:

Particle initial positions file
-------------------------------

Particle release zones
^^^^^^^^^^^^^^^^^^^^^^

A number of tools are provided in the **pylagtools** repository to assist with 
the creation of particle initial positions files. Primarily, these support the
creation of **circular release zones** in which any number of particles can be
placed. Within a given release zone particles can be either regularly or
randomly distributed. The former method is discontinuous, and only the latter
guarantees that the specified number of particles will be created. The two
approaches are illustrated below:

.. plot:: documentation/plots/release_zones.py
   :include-source:

In addition to the creation of individual release zones, PyLag also supports
creating a set of release zones lying along a cord:

.. plot:: documentation/plots/release_zones_along_cord.py
   :include-source:
   
Finally, PyLag also includes support for for creating a set of circular release
zones around an arbitrary body:

.. plot:: documentation/plots/release_zones_around_shape.py
   :include-source:

File format
^^^^^^^^^^^

The file giving the number of particles to be used in the simulation as well as
their initial positions must be given in the run configuration file. The name
is arbitrary, but will typically be something like **initial_positions.dat**. It 
has the following format:

.. literalinclude:: text/initial_positions.dat

The first line gives the number of particles to be used in the simulation. The
remaining lines list each particle's group and its spatial coordinates:

    group_id x y z
    
**group_id** is an integer, and specifies the group to which the particle
belongs. This can be useful when simulating lots of particles released from
different locations. For example, all particles belonging to a given release
zone may be given the same group id. The next three entries are floating point 
values that give the particle's position (**x**, **y**, **z**). **x** and **y** 
are typically given in cartesian coordinates. **z** (positive up) may be given 
in  cartesian or sigma coordinates, and is specified relative to the moving
free surface.

Creating particle initial positions files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyLag includes a small number of methods to help with the creation of initial
positions files. These take ReleaseZone objects as function parameters, extract
from these particle attributes, then write these to file. This is illustrated in
the following code:

.. sourcecode:: python

    from pylagtools import create_release_zone
    from pylagtools import create_initial_positions_file_multi_group

    group_id1 = 1      # Particle group ID #1
    group_id2 = 2      # Particle group ID #2
    n_particles = 10   # No. of particles per release zone
    radius = 100.0     # Radius of each release zone

    release_zones = []
    release_zones.append(create_release_zone(group_id1, radius, n_particles=n_particles))
    release_zones.append(create_release_zone(group_id2, radius, n_particles=n_particles))

    filename = "./initial_positions.dat"
    create_initial_positions_file_multi_group(filename, release_zones)