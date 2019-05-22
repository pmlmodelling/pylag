"""
TODO
----

1) Should get_model() return a statically typed object?
"""

from pylag.model import OPTModel
from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.gotm_data_reader import GOTMDataReader

# Serial imports
from pylag.mediator import SerialMediator

def get_model(config, datetime_start, datetime_end):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        mediator = SerialMediator(config, datetime_start, datetime_end)
        data_reader = FVCOMDataReader(config, mediator)
        return OPTModel(config, data_reader)
    elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "GOTM":
        mediator = SerialMediator(config, datetime_start, datetime_end)
        data_reader = GOTMDataReader(config, mediator)
        return OPTModel(config, data_reader)
    else:
        raise ValueError('Unsupported ocean circulation model.')
