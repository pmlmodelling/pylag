from pylag.model import OPTModel
from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.gotm_data_reader import GOTMDataReader

# Serial imports
from pylag.mediator import SerialMediator

def get_model(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        mediator = SerialMediator(config)
        data_reader = FVCOMDataReader(config, mediator)
        return OPTModel(config, data_reader)
    elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "GOTM":
        mediator = SerialMediator(config)
        data_reader = GOTMDataReader(config, mediator)
        return OPTModel(config, data_reader)
    else:
        raise ValueError('Unsupported ocean circulation model.')