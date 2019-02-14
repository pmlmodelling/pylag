try:
    import configparser
except ImportError:
    import ConfigParser as configparser

def get_config(config_filename=None):
    global _config
    if _config is None:
        # Create new configuration object
        _config = configparser.SafeConfigParser()
        
        if config_filename:
            _config.read(config_filename)
            return _config
        
        # For now, raise an exception if a configuration file was not given.
        # In the future, there may be a scenario when a useful config could be
        # created from defaults alone.
        raise RuntimeError('No configuration file provided.')
        
# ConfigParser
_config = None
