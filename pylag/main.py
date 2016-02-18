import sys
import argparse
import logging
import subprocess

from pylag.configuration import get_config
from pylag.simulator import get_simulator
import pylag.random as random

def main():
    # Parse command line agruments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to the configuration file', metavar='')
    parsed_args = parser.parse_args(sys.argv[1:])

    # Read in run config
    config = get_config(config_filename=parsed_args.config)
    
    # Initiate logging
    logging.basicConfig(filename="{}/pylag_out.log".format(config.get('GENERAL', 'out_dir')),
                        filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=config.get('GENERAL', 'log_level'))
    logger = logging.getLogger(__name__)
    
    # Save the version of the code used (current commit + status)
    logger.info('Starting PyLag')
    logger.info('git log -n -1:\n' + subprocess.check_output(["git", "log", "-n", "1"]))
    logger.info('git status:\n' + subprocess.check_output(["git", "status"]))
    
    # Record configuration to file
    with open("{}/pylag_out.cfg".format(config.get('GENERAL', 'out_dir')), 'wb') as config_out:
        logger.info('Writing run config to file')
        config.write(config_out)
    
    # Initialise the RNG
    random.seed()
    
    # Run the simulation
    simulator = get_simulator(config)
    simulator.run(config)
     
    # End logging and exit
    logger.info('Stopping PyLag')

if __name__ == '__main__':
    main()
