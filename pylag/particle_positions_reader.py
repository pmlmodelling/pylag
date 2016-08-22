import numpy as np

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
    n_particles = _get_entry(data.pop(0)[0], np.int32)

    # Create seed particle set
    group_ids = []
    x_positions = []
    y_positions = []
    z_positions = []
    for row in data:
        group_ids.append(_get_entry(row[0], np.int32))
        x_positions.append(_get_entry(row[1], np.float32))
        y_positions.append(_get_entry(row[2], np.float32))
        z_positions.append(_get_entry(row[3], np.float32))
    
    group_ids = np.array(group_ids)
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    z_positions = np.array(z_positions)
    
    return n_particles, group_ids, x_positions, y_positions, z_positions

def _get_entry(value, type):
    return type(value)