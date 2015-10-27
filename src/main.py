import sys
import argparse
import logging

from configuration import get_config

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
    logger.info('Starting PyLag')
    
    # Record configuration to file
    with open("{}/pylag_out.cfg".format(config.get('GENERAL', 'out_dir')), 'wb') as config_out:
        logger.info('Writing run config to file')
        config.write(config_out)
        
    logger.info('Stopping PyLag')

if __name__ == '__main__':
    main()
