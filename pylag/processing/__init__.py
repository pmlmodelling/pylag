"""
pylag.processing
----------------

The `pylag.processing` subpackage contains modules to assist with
setting up and analysing PyLag simulations. By default, the
dependencies required by the `pylag.processing` subpackage are
not installed with the base code. This is to keep the overall
size of the package small and to make it easier to handle
dependency clashes. To enable the `pylag.processing` subpackage,
please install the extra dependencies listed in the installation
guide: `https://pylag.readthedocs.io/en/latest/`.
"""
from pylag.processing import coordinate
from pylag.processing import release_zone
from pylag.processing import input

# Plotting dependencies
try:
    from pylag.processing import plot
    from pylag.processing import ncview
except (ImportError, ModuleNotFoundError):
    pass

