""" Run a PyLag simulation in parallel

Example
-------

$ mpiexec -np 4 python -m pylag.parallel.main -c pylag.cfg

where the flag `np` is the number of processors, which here has
been set to four.

For additional information on setting up a PyLag simulation,
including creating a new run configuration file, see PyLag's
documentation.

See Also
--------
pylag.main : Run a PyLag simulation in serial
"""

from __future__ import print_function

import os
import sys
import argparse
import logging
import pathlib

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
        parser.add_argument('-c', '--config',
                            help='Path to the configuration file', metavar='')
        parsed_args = parser.parse_args(sys.argv[1:])

        # Read in run config
        try:
            config = get_config(config_filename=parsed_args.config)
        except RuntimeError:
            print('Failed to create run config. Please make sure a config '
                  'file iss given using the -c or --config command line '
                  'arguments.')
            comm.Abort()

        # Create output directory if it does not exist already
        out_dir = config.get('GENERAL', 'out_dir')
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    else:
        config = None

    # Copy the run config to all workers
    config = comm.bcast(config, root=0)    
    
    # Initiate logging
    if rank == 0:
        out_dir = config.get('GENERAL', 'out_dir')
        logging.basicConfig(filename=f"{out_dir}/pylag_out.log",
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=config.get('GENERAL', 'log_level'))
        logger = logging.getLogger(__name__)

        # Save the version of the code used (current commit + status)
        logger.info(f'Starting PyLag-MPI')
        logger.info(f'Using PyLag version: {version.version}')
        logger.info(f'Using {comm.Get_size()} processors')

        # Record configuration to file
        with open(f"{out_dir}/pylag_out.cfg", 'w') as config_out:
            logger.info('Writing run config to file')
            config.write(config_out)

    # Seed the random number generator
    random.seed()
    if config.get('GENERAL', 'log_level') == 'DEBUG':
        print(f'Random seed for processor with rank {rank} is '
              f'{random.get_seed()}')
    
    # Run the simulation
    simulator = get_simulator(config)
    simulator.run()
    
    # End logging and exit
    if rank == 0:
        logger.info('Stopping PyLag')


if __name__ == '__main__':
    main()
