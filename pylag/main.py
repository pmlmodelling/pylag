"""
Run a PyLag simulation in serial

Example
-------

$ python -m pylag.main -c pylag.cfg

For additional information on setting up a PyLag simulation,
including creating a new run configuration file, see PyLag's
documentation.

See Also
--------
pylag.parallel.main : Run a PyLag simulation in parallel
"""

from __future__ import print_function

import os
import sys
import argparse
import logging
import pathlib

from pylag.configuration import get_config
from pylag.simulator import get_simulator
from pylag.exceptions import PyLagRuntimeError
import pylag.random as random
from pylag import version


def main():
    # Parse command line agruments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help='Path to the configuration file', metavar='')
    parsed_args = parser.parse_args(sys.argv[1:])

    # Read in run config
    try:
        config = get_config(config_filename=parsed_args.config)
    except RuntimeError as re:
        print('Failed to create run config. Please make sure a config '
              'file is given using the -c or --config command line '
              'arguments.')
        raise PyLagRuntimeError(re.message)

    # Create output directory if it does not exist already
    out_dir = config.get('GENERAL', 'out_dir')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
    
    # Initiate logging
    logging.basicConfig(filename=f"{out_dir}/pylag_out.log",
                        filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=config.get('GENERAL', 'log_level'))
    logger = logging.getLogger(__name__)
    
    # Save the version of the code used (current commit + status)
    logger.info('Starting PyLag')
    logger.info(f"Using PyLag version: {version.version}")
    
    # Record configuration to file
    with open(f"{out_dir}/pylag_out.cfg", 'w') as config_out:
        logger.info('Writing run config to file')
        config.write(config_out)
    
    # Initialise the RNG
    random.seed()
    
    # Run the simulation
    simulator = get_simulator(config)
    simulator.run()
     
    # End logging and exit
    logger.info('Stopping PyLag')


if __name__ == '__main__':
    main()
