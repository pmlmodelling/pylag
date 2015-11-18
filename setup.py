# distutils: define_macros=CYTHON_TRACE_NOGIL=1

from distutils.core import setup
from Cython.Build import cythonize

setup(name="PyLag",
      version="X.X",
      description="Python/Cython Lagrangian Modelling Frmaework",
      author="Jame Clark (PML)",
      author_email="jcl@pml.ac.uk",
      url="TODO",
      ext_modules=cythonize(["time_manager.pyx", "particle.pyx", "fvcom_data_reader.pyx", "integrator.pyx"]),
)
