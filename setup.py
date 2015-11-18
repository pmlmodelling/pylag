# distutils: define_macros=CYTHON_TRACE_NOGIL=1

from distutils.core import setup
from Cython.Build import cythonize

setup(name="pylag",
      version="X.X",
      description="Python/Cython Lagrangian Modelling Frmaework",
      author="Jame Clark (PML)",
      author_email="jcl@pml.ac.uk",
      url="gitlab.ecosystem-modelling.pml.ac.uk:jimc/PyLag.git",
      packages=["pylag"],
      ext_modules=cythonize(["pylag/*.pyx"], include_path = ["./include"]),
)
