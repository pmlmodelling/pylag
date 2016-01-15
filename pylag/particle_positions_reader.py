import logging

def read_particle_initial_positions(file_name):
    """
    Read in particle initial positions from file and return to caller.
    
    Parameters
    ----------
    file_name: string
        Name of the file.
        
    Returns
    -------
    group_id: list, int
        List of particle group ids
    xpos: list, float
        List of x position
    ypos: list, float
        List of y positions
    zpos: list, float
        List of z positions (either cartesian or sigma coordinates)
    """
    
    # Input file containing particle initial positions
    with open(file_name, 'r') as f:
        lines=f.readlines()

    data=[]
    for line in lines:
        data.append(line.rstrip('\r\n').split(' '))    
        
    # The first entry is the number of particles
    n_particles = _get_entry(data.pop(0)[0], int)

    # Create seed particle set
    group_id = []
    xpos = []
    ypos = []
    zpos = []
    for row in data:
        group_id.append(_get_entry(row[0], int))
        xpos.append(_get_entry(row[1], float))
        ypos.append(_get_entry(row[2], float))
        zpos.append(_get_entry(row[3], float))

    logger = logging.getLogger(__name__)
    if n_particles == len(group_id):
        logger.info('Particle seed contains {} particles.'.format(n_particles))
        return group_id, xpos, ypos, zpos
    else:
        logger.warning('Error reading particle initial positions from file. '\
            'The number of particles specified in the file is {}. The actual number found '\
            'while parsing the file was {}.'.format(n_particles, len(_seed)))
        raise ValueError('Error reading particle initial positions file. See log ' \
            'file for more information.')

def _get_entry(value, type):
    return type(value)