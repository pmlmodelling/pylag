"""
Module containing factory method for Model Objects in parallel runs

See Also
--------
pylag.model_factory - Factory method for serial execution
"""
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.model import OPTModel
from pylag.arakawa_a_data_reader import ArakawaADataReader
from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.roms_data_reader import ROMSDataReader
from pylag.gotm_data_reader import GOTMDataReader
from pylag.atmosphere_data_reader import AtmosphereDataReader
from pylag.waves_data_reader import WavesDataReader
from pylag.composite_data_reader import CompositeDataReader
from pylag.exceptions import PyLagValueError

# Serial imports
from pylag.parallel.mediator import MPIMediator


def get_model(config, datetime_start, datetime_end):
    """ Factory method for model objects.

    Configure the OPTModel. Here, the required set of data readers
    for ocean, wind and waves input data are instantiated and
    configured. Tests are put in place to ensure a working
    configuration has been specified.

    Parameters
    ----------
    config : ConfigParser
        Configuration object

    datetime_start : Datetime
        Start datetime

    datetime_end : Datetime
        End datetime

    Returns
    -------
     : pylag.model.OPTModel
         Offline particle tracking model object
    """
    # Ocean data
    # ----------
    try:
        ocean_product_name = config.get("OCEAN_CIRCULATION_MODEL",
                                        "name").strip()
    except (configparser.NoSectionError, configparser.NoOptionError):
        ocean_product_name = "none"

    if ocean_product_name.lower() != "none":
        data_source = 'ocean'
        ocean_mediator = MPIMediator(config, data_source, datetime_start,
                                     datetime_end)
        if ocean_product_name == "ArakawaA":
            ocean_data_reader = ArakawaADataReader(config, ocean_mediator)
        elif ocean_product_name == "FVCOM":
            ocean_data_reader = FVCOMDataReader(config, ocean_mediator)
        elif ocean_product_name == "ROMS":
            ocean_data_reader = ROMSDataReader(config, ocean_mediator)
        elif ocean_product_name == "GOTM":
            ocean_data_reader = GOTMDataReader(config, ocean_mediator)
        else:
            raise PyLagValueError(f"Unsupported ocean data product "
                                  f"`{ocean_product_name}`")
    else:
        ocean_data_reader = None

    # Atmosphere data
    # ---------------
    try:
        atmos_product_name = config.get("ATMOSPHERE_DATA", "name").strip()
    except (configparser.NoSectionError, configparser.NoOptionError):
        atmos_product_name = "none"

    if atmos_product_name.lower() != "none":
        data_source = 'atmosphere'
        atmos_mediator = MPIMediator(config, data_source, datetime_start,
                                     datetime_end)
        if atmos_product_name == "Default":
            atmos_data_reader = AtmosphereDataReader(config, atmos_mediator)
        else:
            raise PyLagValueError(f"Unsupported atmosphere data product "
                                  f"`{atmos_product_name}`")
    else:
        atmos_data_reader = None

    # Waves data
    # ----------
    try:
        waves_product_name = config.get("WAVE_DATA", "name").strip()
    except (configparser.NoSectionError, configparser.NoOptionError):
        waves_product_name = "none"

    if waves_product_name.lower() != "none":
        data_source = 'wave'
        waves_mediator = MPIMediator(config, data_source, datetime_start,
                                     datetime_end)
        if waves_product_name == "Default":
            waves_data_reader = WavesDataReader(config, waves_mediator)
        else:
            raise PyLagValueError(f"Unsupported waves data product "
                                  f"`{waves_product_name}`")
    else:
        waves_data_reader = None

    # Error checking
    # --------------
    if ocean_data_reader is None and atmos_data_reader is None and \
            waves_data_reader is None:
        raise PyLagValueError(f"To build an OPTModel, at least one source "
                              f"of input data must be specified.")

    if ocean_data_reader is None and atmos_data_reader is not None and \
            waves_data_reader is not None:
        raise PyLagValueError(f"At the current time, PyLag does not "
                              f"support running the model with atmosphere "
                              f"and waves data without ocean data")

    # Build the OPTModel
    # ------------------
    if ocean_data_reader is not None and atmos_data_reader is None and \
            waves_data_reader is None:
        opt_model = OPTModel(config, ocean_data_reader)
    elif ocean_data_reader is None and atmos_data_reader is not None and \
            waves_data_reader is None:
        opt_model = OPTModel(config, atmos_data_reader)
    elif ocean_data_reader is None and atmos_data_reader is None and \
            waves_data_reader is not None:
        opt_model = OPTModel(config, waves_data_reader)
    else:
        # Using a combination of data types - requires a composite data reader
        composite_data_reader = CompositeDataReader(config,
                                                    ocean_data_reader,
                                                    atmos_data_reader,
                                                    waves_data_reader)
        opt_model = OPTModel(config, composite_data_reader)

    return opt_model


__all__ = ['get_model']
