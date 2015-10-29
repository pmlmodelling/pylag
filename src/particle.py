import numpy as np
import logging

class Particle(object):
    def __init__(self, group_id, xpos, ypos, zpos):
        self._group_id = group_id
        
        self._xpos = xpos
        self._ypos = ypos
        self._zpos = zpos
    
    # x position
    @property
    def xpos(self):
        return self._xpos
    
    @xpos.setter
    def xpos(self, value):
        self._xpos = value
    
    # y postion
    @property
    def ypos(self):
        return self._ypos
    
    @ypos.setter
    def ypos(self, value):
        self._ypos = value
    
    # z position
    @property
    def zpos(self):
        return self._zpos

    @zpos.setter
    def zpos(self, value):
        self._zpos = value

def get_particle_seed(config=None):
    global _seed
    
    # Return a copy of the particle seed if it has already been created
    if _seed:
        return _seed
    
    # Input file containing particle initial positions
    file_name = config.get('GENERAL', 'initial_positions_file')
    with open(file_name, 'r') as f:
        lines=f.readlines()

    data=[]
    for line in lines:
        data.append(line.rstrip('\r\n').split(' '))    
        
    # The first entry is the number of particles
    n_particles = _get_entry(data.pop(0)[0], int)

    # Create seed particle set
    _seed = []
    for row in data:
        group_id = _get_entry(row[0], int)
        xpos = _get_entry(row[1], float)
        ypos = _get_entry(row[2], float)
        zpos = _get_entry(row[3], float)
        particle = Particle(group_id, xpos, ypos, zpos)
        _seed.append(particle)

    logger = logging.getLogger(__name__)
    if n_particles == len(_seed):
        logger.info('Particle seed contains {} particles.'.format(n_particles))
        return np.copy(_seed)
    else:
        logger.warning('Error reading particle initial positions from file. '\
            'The number of particles specified in the file is {}. The actual number found '\
            'while parsing the file was {}.'.format(n_particles, len(_seed)))
        raise ValueError('Error reading particle initial positions file. See log ' \
            'file for more information.')

def _get_entry(value, type):
    return type(value)

# Initial particle set with particle starting positions initalised from file
_seed = None
