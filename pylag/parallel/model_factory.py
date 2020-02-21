from pylag.model import OPTModel
from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.gotm_data_reader import GOTMDataReader

# Parallel imports
from pylag.parallel.mediator import MPIMediator


def get_model(config, datetime_start, datetime_end):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "ArakawaA":
        mediator = MPIMediator(config, datetime_start, datetime_end)
        data_reader = ArakawaADataReader(config, mediator)
        return OPTModel(config, data_reader)
    elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        mediator = MPIMediator(config, datetime_start, datetime_end)
        data_reader = FVCOMDataReader(config, mediator)
        return OPTModel(config, data_reader)
    elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "GOTM":
        mediator = MPIMediator(config, datetime_start, datetime_end)
        data_reader = GOTMDataReader(config, mediator)
        return OPTModel(config, data_reader)
    else:
        raise ValueError('Unsupported ocean circulation model.')
