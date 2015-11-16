# distutils: define_macros=CYTHON_TRACE_NOGIL=1

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["particle.pyx", "time_manager.pyx", "model_reader.pyx"]),
)
