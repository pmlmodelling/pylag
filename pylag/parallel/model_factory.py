from pylag.model import FVCOMOPTModel
from pylag.fvcom_data_reader import FVCOMDataReader

# Parallel imports
from pylag.parallel.mediator import FVCOMMediator

def get_model(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        mediator = FVCOMMediator(config)
        data_reader = FVCOMDataReader(config, mediator)
        return FVCOMOPTModel(config, data_reader)
    else:
        raise ValueError('Unsupported ocean circulation model.')