# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import os
import subprocess
import glob
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

MAJOR               = 0
MINOR               = 1
ISRELEASED          = False
VERSION = '{}.{}'.format(MAJOR, MINOR)

build_type = 'prod'
#build_type = 'prof'
#build_type = 'debug'

def git_version():
    """Return the git revision as a string (adapted from NUMPY source)

    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C++'
        env['LANG'] = 'C++'
        env['LC_ALL'] = 'C++'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def get_version_info():
    """ Return version info (adapted from NUMPY source) """
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pylag/version.py'):
        # must be a source distribution, use existing version file
        try:
            from pylag.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing " \
                              "pylag/version.py and the build directory " \
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION

def write_version_py(filename='pylag/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM PyLag SETUP.PY
#
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    with open(filename, 'w') as a:
        try:
            a.write(cnt % {'version': VERSION,
                           'full_version' : FULLVERSION,
                           'git_revision' : GIT_REVISION,
                           'isrelease': str(ISRELEASED)})
        finally:
            pass

def scandir(dir, file_type, files=[]):
    """ Scan the directory for extension files

        Convert these to extension names in dotted notation

        Adapted from Cython wiki
    """
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(file_type):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, file_type, files)
    return files


def makeExtension(ext_name, file_type):
    """ Generate an Extension object from its dotted name (adapted
        from Cython wiki).

        If the extension object is wrapping a C++ source file, then
        we must append its name too. In order to find these files
        automatically, we have adopted the following naming
        convention:

        <module_name>.h
        <module_name>.cpp
        <module_name>_cpp_wrapper.pyx
    """
    ext_path = ext_name.replace(".", os.path.sep)+file_type

    cpp_path = None
    substring = '_cpp_wrapper'
    if substring in ext_path:
        i = ext_path.find(substring)
        stem = ext_path[:i]
        cpp_path = stem + '.cpp'

    ext_path_list = [ext_path] if cpp_path is None else [ext_path, cpp_path]

    return Extension(
        ext_name,
        ext_path_list,
        language="c++",
        extra_compile_args=["-std=c++11"],
        libraries=["stdc++"],
        include_dirs=['.'],
        )

# Rewrite the version file everytime
write_version_py()

# Get a list of source files
file_type = '.pyx' if os.path.isdir('./.git') and glob.glob('./pylag/*.pyx') else '.cpp'

# Get the list of extensions
ext_names = scandir("pylag", file_type)

# And build up the set of Extension objects
extensions = [makeExtension(name, file_type) for name in ext_names]

# Cythonize if working with pyx files
if file_type == '.pyx':
    if build_type == 'prod':
        ext_modules = cythonize(extensions, include_path=['include'],
              compiler_directives={'boundscheck': False})
    elif build_type == 'prof':
        ext_modules = cythonize(extensions, include_path=['include'],
              compiler_directives={'profile': True, 'linetrace': True})
    elif build_type == 'debug':
        ext_modules = cythonize(extensions, include_path=['include'],
              compiler_directives={'profile': True, 
              'linetrace': True, 'boundscheck': True,
              'cdivision_warnings': True, 'initializedcheck': True,
              'nonecheck': True, 'embedsignature': True
              }, gdb_debug=True, verbose=True)
    else:
        raise ValueError('Unknown build_type {}'.format(build_type))
elif file_type == '.cpp':
    ext_modules = extensions

setup(name="PyLag",
      version=get_version_info()[0],
      description="Python/Cython Lagrangian Modelling Frmaework",
      author="Jame Clark (PML)",
      author_email="jcl@pml.ac.uk",
      url="gitlab.ecosystem-modelling.pml.ac.uk:jimc/PyLag.git",
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Intended Audience :: Scientific researchers',
          'Topic :: Particle Tracking',
          'Programming Language :: Python :: 2.7',
          'Operating System :: UNIX',
      ],
      packages=["pylag", "pylag.parallel"],
      ext_modules=ext_modules,
      install_requires=[
          "cython",
          "numpy",
          "netCDF4",
          "mpi4py",
          "ConfigParser2",
          "natsort",
          "progressbar",
          "nose",
          "sphinx_rtd_theme",
      ],
)
