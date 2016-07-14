# distutils: define_macros=CYTHON_TRACE_NOGIL=1
import os
import glob
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import cython_gsl

# scan the directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, file_type, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(file_type):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


# generate an Extension object from its dotted name
def makeExtension(extName, file_type):
    extPath = extName.replace(".", os.path.sep)+file_type
    return Extension(
        extName,
        [extPath],
        libraries=cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir()],
        include_dirs=[cython_gsl.get_cython_include_dir(), '.'],
        )

# Get a list of source files
file_type = '.pyx' if os.path.isdir('./.git') and glob.glob('./pylag/*.pyx') else '.c'

# Get the list of extensions
extNames = scandir("pylag", file_type)

# And build up the set of Extension objects
extensions = [makeExtension(name, file_type) for name in extNames]

# Cythonize if working with pyx files
if file_type == '.pyx':
    ext_modules = cythonize(extensions, include_path=['include'])
elif file_type == '.c':
    ext_modules = extensions

setup(name="PyLag",
      version="0.1",
      description="Python/Cython Lagrangian Modelling Frmaework",
      author="Jame Clark (PML)",
      author_email="jcl@pml.ac.uk",
      url="gitlab.ecosystem-modelling.pml.ac.uk:jimc/PyLag.git",
      license='MIT',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Intended Audience :: Scientific researchers',
          'Topic :: Particle Tracking',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Operating System :: UNIX',
      ],
      packages=["pylag"],
      ext_modules=extensions,
      #ext_modules=cythonize(extensions, include_path=['include'],
      #    compiler_directives={'profile': True, 
      #    'linetrace': True, 'boundscheck': True,
      #    'cdivision_warnings': True, 'initializedcheck': True}),
)
