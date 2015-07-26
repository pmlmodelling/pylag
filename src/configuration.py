import ConfigParser

def get_config(config_filename=None):
    global _config, _defaults
    if _config is None:
        # Create new configuration object
        _config = ConfigParser.SafeConfigParser(_defaults)
        
        if config_filename: _config.read(config_filename)
    return _config

# ConfigParser
_config = None

# Default configuration options
_defaults = {'log_level': 'INFO', 'in_dir': './', 'out_dir': './'}
