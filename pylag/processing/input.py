"""
Tools to assist with the creation of PyLag initial position files
"""

from __future__ import division, print_function


def create_initial_positions_file_single_group(filename, n, group_id, xpos, ypos, zpos):
    """ Create an initial positions file for a single group of particles

    Create a file specifying the initial positions of a set of n particles. Each
    particle has the same group ID.

    Expected file format is: \n
    n \n
    group_id_1 xpos_1 ypos_1 zpos_1 \n
    group_id_1 xpos_2 ypos_2 zpos_2 \n
    ... \n
    ... \n
    group_id_2 xpos_n ypos_n zpos_n \n

    where n is the total number of particles, {x,y,z}pos_i is the initial
    x/y/z position of particle i (i= 1 to n) in UTM coordinates, and 
    group_id_1 is an integer specifying the group to which the particle belongs.

    Parameters
    ----------
    filename : string
        Output file name (either local or full path)

    n : integer
        Total number of particles to be released.

    group_id : integer
        Group id for all particles.

    xpos : ndarray
        UTM x coordinate position of the particle.

    ypos : ndarray
        UTM y coordinate position of the particle.

    zpos : ndarray
        Particle depth.
    """ 
    if n != len(xpos) or n != len(ypos) or n != len(zpos):
        raise ValueError('Number of particles does not match the '\
                'number of spatial coordinates provided.')
    
    f = open(filename, 'w')
    f.write(str(n) + '\n')
    for x, y, z in zip(xpos, ypos, zpos):
        line = str(group_id) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
        f.write(line)
    f.close()


def create_initial_positions_file_multi_group(filename, release_zones):
    """ Create an initial positions file for multiple particle groups

    Take a list of ReleaseZone objects, extract particle initial positions and
    write to file.

    Expected file format is: \n
    n \n
    group_id_1 xpos_1 ypos_1 zpos_1 \n
    group_id_1 xpos_2 ypos_2 zpos_2 \n
    ... \n
    ... \n
    group_id_2 xpos_n ypos_n zpos_n \n

    where n is the total number of particles, {x,y,z}pos_i is the initial
    x/y/z position of particle i (i= 1 to n) in UTM coordinates, and 
    group_id_1 is an integer specifying the group to which the particle belongs.

    Parameters
    ----------
    filename : string
        Output file name (either local or full path)

    release_zones : ReleaseZone, iterable
        List or array of release zone objects each containing an arbitrary number 
        of particles.
    """

    # Total number of particles across all release zones
    n = 0
    for release_zone in release_zones:
        n = n + release_zone.get_number_of_particles()

    f = open(filename, 'w')
    f.write(str(n) + '\n')
    for release_zone in release_zones:
        x_arr, y_arr, z_arr = release_zone.get_coordinates()
        for x, y, z in zip(x_arr, y_arr, z_arr):
            line = (str(release_zone.group_id) +
                    ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
            f.write(line)
    f.close()
