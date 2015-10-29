class ModelFactory(object):
    def __init__(self, config):
        self.config = config
    
    def make_grid_reader(self): pass

class FVCOMModelFactory(ModelFactory):
    def __init__(self, *args, **kwargs):
        super(FVCOMModelFactory, self).__init__(*args, **kwargs)

    def make_grid_reader(self):
        pass

def get_model_factory(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "NAME") == "FVCOM":
        return FVCOMModelFactory(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')    
