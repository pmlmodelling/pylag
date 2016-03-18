# distutils: define_macros=CYTHON_TRACE_NOGIL=1

from distutils.core import setup, Extension
from Cython.Build import cythonize
import cython_gsl

extensions = [
    Extension("*", 
        ["pylag/*.pyx"],
        libraries = cython_gsl.get_libraries(),
        library_dirs = [cython_gsl.get_library_dir()],
        include_dirs = ["./include", cython_gsl.get_cython_include_dir()])      
]

setup(name="PyLag",
      version="X.X",
      description="Python/Cython Lagrangian Modelling Frmaework",
      author="Jame Clark (PML)",
      author_email="jcl@pml.ac.uk",
      url="gitlab.ecosystem-modelling.pml.ac.uk:jimc/PyLag.git",
      packages=["pylag"],
      ext_modules=cythonize(extensions),
)
