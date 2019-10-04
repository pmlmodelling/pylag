# Updated 4th October 2019

# PyLag

PyLag is an offline particle tracking model. The model expects as inputs time independent and/or dependent variables that describe the state of a given fluid. These may be measured quantities or the predictions of an analytical or numerical model. Using these, the model computes Lagrangian trajectories for particles released into the fluid at a particular point in time and space. The model is primarily aimed at marine applications, but is intended to be versatile enough to be extended to other contexts; for example, studies of atmospheric dispersion.

PyLag was created with the aim to make available a particle tracking model that is a) fast to run, b) easy to use, c) extensible and d) flexible. The model is written in a mixture of [Python](http://www.python.org) and [Cython](http://www.cython.org).

## Installation

The easiest way to install *PyLag* is using *Conda*, which will install *PyLag* and all its dependencies into a clean environment. However, it is also possible to install *PyLag* using *pip*, assuming your environment has been correctly configured. Both approaches are described below. For now, only linux-based platforms are supported.

### Installation using Conda


---
**NOTE**

The instructions below assume you are working with *Python 3*; however, *PyLag* supports both *Python 2* and *Python 3*.

---

First [install miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) in a location of your choosing. Then, activate *conda*; add the *conda-forge* channel; and install *conda-build* and *conda-verify*, which we will use to install *PyLag*. For example:

```bash
$ source /opt/miniconda/miniconda3/bin/activate
$ conda config --append channels conda-forge
$ conda install conda-build conda-verify
```

The above code will install *miniconda3* into the directory `/opt/miniconda`, once the appropriate write permissions have been set (the default is to install *miniconda3* into your home directory, which is, of course, also fine).

With *miniconda3* installed and configured, create a new environment in which to install *PyLag* using the following commands:

```bash
$ conda create -n particles
$ conda activate particles
```

Next, clone *PyLag* using *git*. The code below assumes that you have configured *ssh* access to *GitLab*:

```bash
$ mkdir -p $HOME/code/git/PyLag && cd $HOME/code/git/PyLag
$ git clone git@gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag.git
$ cd PyLag
```

Finally, build and install *PyLag* using conda:

```bash
$ conda build .
$ conda install -n particles --use-local pylag
```

### Installation using pip

The cleanest way to install *PyLag* using *pip* is by using  [virtualenv](https://virtualenv.pypa.io/en/stable/) to create a new virtual environment, then using pip to install *PyLag* within the new virtual environment. Alternatively, PyLag can be installed locally by passing the `--user` flag to *pip*. However, this method may create issues with system or local package dependencies -- something *virtualenv* is designed to avoid. The use of *sudo* -- which would allow *PyLag* to be installed at the system level -- is strongly discouraged.

While *virtualenv* should smoothly handle *PyLag's* python dependencies, it cannot help with non-python packages required by *PyLag*. These include a *C++* compiler, and *MPI*. If you hit problems with these, perhaps try installing *PyLag* using *Conda* instead.

To begin installing *PyLag* using *pip*, use [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to configure a new virtual environment:

```bash
$ mkvirtualenv -p /usr/bin/python3 --no-site-packages pylag
```

Python includes a dependency on the python package [MPI for Python](https://mpi4py.readthedocs.io/en/stable/), which is used to facilitate running particle tracking simulations in parallel. To install *MPI for Python*, it is first necessary to ensure that you have a working MPI implementation on your system, and that all paths to MPI libraries and header files have been correctly set. You must use your Linux package manager to install a working MPI Implementation. On my laptop running Fedora 27, the following commands suffice:

```bash
$ sudo dnf install -y mpich mpich-devel python3-mpich
```

Alternatively, if it is available via your package manager, you can install *python3-mpi4py* at the system level, which should automatically install all necessary MPI dependencies.

On my machine, *mpich* is enabled using the module command, which correctly sets environmental paths to the *mpich* MPI libraries and header files:

```bash
$ module load mpi/mpich-x86_64
```
---
**NOTE**

If this fails, try using `module avail` to list available MPI modules.

---

With MPI support enabled, it is now possible to install *PyLag* within the new virtual environment:

```bash
$ mkdir -p $HOME/code/git/PyLag && cd $HOME/code/git/PyLag
$ git clone git@gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag.git
$ cd PyLag
$ pip install . 
```

To test that *PyLag* has been correctly installed, type:

```bash
$ python -c "import pylag"
```

which should exit without error.

To install *PyLag* locally is arguably easier, since if *MPI for Python* is installed at the system level, you will not have to worry about updating your paths for MPI (this is required in the virtual environment, since *MPI for Python* is built from source after being pulled down from *PyPI*).If you do install *PyLag* locally, simply pass the `--user` flag to both invocations of `pip install`.

Notes
=====

- There is no current support for compiling PyLag with Intel compilers (GCC works OK).

- The C compiler must support the C++11 standard (i.e. GCC versions >= 4.8, Intel versions >= 12).

