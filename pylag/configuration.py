"""
PyLag runtime configuration options are set in a dedicated run configuration
file, which typically has a name like "pylag.cfg". The run configuration file
is formatted for configparser, which is used here to read in the configuration
file. A object of type ConfigParser is saved as a module level variable
with a reference returned to the caller. This structure allows the same
ConfigParser to be passed to all PyLag objects that use information within
the PyLag configuration file during class initialisation.
"""

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


def get_config(config_filename=None):
    """ Get the run config

    When called for the first time, the config file name must
    be provided. It will then be opened and parsed using
    `configparser`. The resulting `ConfigParser` is saved as
    a private, module-level variable. Subsequent calls to
    this function can be used to obtain a reference to the
    ConfigParser.

    Parameters
    ----------
    config_filename : str or None, optional
        The name of the run configuration file including its full or
        relative path. This is only optional if the `ConfigParser`
        has been created during a previous call to `get_config`.

    Returns
    -------
    _config : ConfigParser
        The ConfigParser.

    Raises
    ------
    RuntimeError
        Tried to call `get_config` for the first time without a file
        name.

    """
    global _config
    if _config is None:
        # Create new configuration object
        _config = configparser.ConfigParser()
        
        if config_filename:
            _config.read(config_filename)
            return _config
        
        # For now, raise an exception if a configuration file was not given.
        # In the future, there may be a scenario when a useful config could be
        # created from defaults alone.
        raise RuntimeError('No configuration file provided.')

    return _config


# ConfigParser
_config = None
