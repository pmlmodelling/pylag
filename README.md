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



Notes
=====

- Currently there appears to be an incompatibility with Cython versions > 0.25.

- There is no current support for compiling PyLag with Intel compilers (GCC works OK).

- The C compiler must support the C++11 standard (i.e. GCC versions >= 4.8, Intel versions >= 12).

