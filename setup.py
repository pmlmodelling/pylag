# distutils: define_macros=CYTHON_TRACE_NOGIL=1
import os
import subprocess
import glob
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import cython_gsl

MAJOR               = 0
MINOR               = 1
ISRELEASED          = False
VERSION = '{}.{}'.format(MAJOR, MINOR)

# Return the git revision as a string (adapted from NUMPY source)
def git_version():
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

# Return version info (adapted from NUMPY source)
def get_version_info():
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

# Scan the directory for extension files, converting
# them to extension names in dotted notation (adapted
# from Cython wiki)
def scandir(dir, file_type, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(file_type):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, file_type, files)
    return files


# generate an Extension object from its dotted name (adapted
# from Cython wiki)
def makeExtension(extName, file_type):
    extPath = extName.replace(".", os.path.sep)+file_type
    return Extension(
        extName,
        [extPath],
        language="c++",
        libraries=["stdc++"] + cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir()],
        include_dirs=[cython_gsl.get_cython_include_dir(), '.'],
        )

# Get a list of source files
file_type = '.pyx' if os.path.isdir('./.git') and glob.glob('./pylag/*.pyx') else '.cpp'

# Get the list of extensions
extNames = scandir("pylag", file_type)

# And build up the set of Extension objects
extensions = [makeExtension(name, file_type) for name in extNames]

# Cythonize if working with pyx files
if file_type == '.pyx':
    ext_modules = cythonize(extensions, include_path=['include'])#,
          #compiler_directives={'profile': True, 
          #'linetrace': True, 'boundscheck': True,
          #'cdivision_warnings': True, 'initializedcheck': True})
elif file_type == '.cpp':
    ext_modules = extensions

# Rewrite the version file everytime
write_version_py()

setup(name="PyLag",
      version=get_version_info()[0],
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
      packages=["pylag", "pylag.parallel"],
      ext_modules=extensions,
)
