import os
import sys
import argparse
import logging

from mpi4py import MPI

from pylag.configuration import get_config
import pylag.random as random
from pylag import version

from pylag.parallel.simulator import get_simulator

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Master (rank=0) controls run initialisation
    if rank == 0:
        # Parse command line agruments
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', help='Path to the configuration file', metavar='')
        parsed_args = parser.parse_args(sys.argv[1:])

        # Read in run config
        try:
            config = get_config(config_filename=parsed_args.config)
        except RuntimeError:
            print 'Failed to create run config. Please make sure a config '\
                'file iss given using the -c or --config command line '\
                'arguments.'
            comm.Abort()
            
        # Create output directory if it does not exist already
        if not os.path.isdir('{}'.format(config.get('GENERAL', 'out_dir'))):
            os.mkdir('{}'.format(config.get('GENERAL', 'out_dir')))
    else:
        config = None

    # Copy the run config to all workers
    config = comm.bcast(config, root=0)    
    
    # Initiate logging
    if rank == 0:
        logging.basicConfig(filename="{}/pylag_out.log".format(config.get('GENERAL', 'out_dir')),
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=config.get('GENERAL', 'log_level'))
        logger = logging.getLogger(__name__)

        # Save the version of the code used (current commit + status)
        logger.info('Starting PyLag-MPI')
        logger.info('Using PyLag version: {}'.format(version.version))
        logger.info('Using {} processors'.format(comm.Get_size()))

        # Record configuration to file
        with open("{}/pylag_out.cfg".format(config.get('GENERAL', 'out_dir')), 'wb') as config_out:
            logger.info('Writing run config to file')
            config.write(config_out)

    # Seed the random number generator
    random.seed()
    if config.get('GENERAL', 'log_level') == 'DEBUG':
        print 'Random seed for pocessor with rank {} is {}'.format(rank, random.get_seed())
    
    # Run the simulation
    simulator = get_simulator(config)
    simulator.run(config)
    
    # End logging and exit
    if rank == 0:
        logger.info('Stopping PyLag')

if __name__ == '__main__':
    main()
