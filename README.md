# Updated 24th April 2020

PyLag is an offline particle tracking model. The model expects as inputs time independent and/or dependent variables that describe the state of a given fluid. These may be measured quantities or the predictions of an analytical or numerical model. Using these, the model computes Lagrangian trajectories for particles released into the fluid at a particular point in time and space. The model is primarily aimed at marine applications, but is intended to be versatile enough to be extended to other contexts; for example, studies of atmospheric dispersion.

PyLag was created with the aim to make available a particle tracking model that is a) fast to run, b) easy to use, c) extensible and d) flexible. The model is written in a mixture of [Python](http://www.python.org) and [Cython](http://www.cython.org).

## How to cite

Uncles, R. J., Clark, J. R., Bedington, M., Torres, R. 2020. “On sediment dispersal in the Whitsand Bay Marine Conservation Zone: Neighbor to a closed dredge-spoil disposal site” in Marine Protected Areas: Evidence, Policy and Practise, ed Robert Clark and John Humphreys (Elsevier Inc.).

## Getting started

For end users, the easiest way to install *PyLag* is using *Conda*. This will install *PyLag* and all its dependencies into a clean environment and make it available to use. Developers who wish to work with the source code directly should clone a copy of *PyLag's* source code and install it manually. Both use cases are described below.

---
**NOTE**

*PyLag* has been developed and tested within a UNIX/Linux environment, and the following instructions assume the user is working within a similar environment.

---

### Instructions for end users

First [install miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) in a location of your choosing. Then, activate *Conda* and add the channels *conda-forge* and *JimClark*. The latter is a temporary distribution channel for *PyLag*. For example:

```bash
$ source /opt/miniconda/miniconda3/bin/activate
(base) $ conda config --append channels conda-forge
(base) $ conda config --append channels JimClark
```

The above code assumes *miniconda3* has been installed in the directory `/opt/miniconda`, once the appropriate write permissions have been set (the default is to install *miniconda3* into your home directory, which is, of course, also fine).

With *miniconda3* installed and configured, create a new environment in which to install *PyLag* using the following commands:

```bash
(base) $ conda create -n particles python=3.7
(base) $ conda activate particles
```

Finally, install *PyLag*:

```bash
(particles) $ conda install -n particles -c JimClark pylag
```

To test that *PyLag* has been correctly installed, type:

```
(particles) $ python -c "import pylag"
```

which should exit without error.

### Instructions for developers

With SSH access setup, you can clone the PyLag repository using the following commands:

```bash
$ mkdir -p $HOME/code/git/PyLag && cd $HOME/code/git/PyLag
$ git clone https://gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag.git>
```

If you don't want to use git to access the code, you can always grab a copy by downloading and unpacking tarballs of the two repositories. The cleanest and safest way of installing *PyLag's* dependencies is using *Conda*. Following steps similar to those described above, we can configure a new *Conda* environment so:

```bash
$ source /opt/miniconda/miniconda3/bin/activate
(base) $ conda config --append channels conda-forge
(base) $ conda config --append channels JimClark
(base) $ conda install conda-build conda-verify
```

The only new step here is the installation of conda-build and conda-verify. Next, create a new environment as above:

```bash
(base) $ conda create -n particles python=3.7
(base) $ conda activate particles
```

And finally, in the PyLag source code directory, build and install *PyLag*.

```bash
(particles) $ cd $HOME/code/git/PyLag/PyLag
(particles) $ conda build .
(particles) $ conda install -n particles --use-local pylag
```

Occsionally, when building *PyLag* this way, users have hit upon clashes with locally installed packages. To get around this problem, you may find it useful to add the following aliases to your bashrc file, which you can use to activate and deactivate *Conda*:

```bash
alias start_conda='export PYTHONNOUSERSITE=True && source /opt/miniconda/miniconda3/bin/activate'
alias stop_conda='unset PYTHONNOUSERSITE && conda deactivate'
```

The *Conda* build process is quite long, and it doesn't lend itself to rapid build-install-test cycles. If you find yourself wanting to perform repeated builds, it is recommended you build using *Conda* on the first attempt, as described above. This will ensure PyLag's dependencies are correctly installed. Then, after this, you can install *PyLag* using *pip* like so:

```bash
(particles) $ cd $HOME/code/git/PyLag/PyLag
(particles) $ pip install .
```

## Further information

[PyLag's documentation](https://drive.google.com/open?id=1Qp5Z_IihcHRpbehDyWfaCofrJ84lJDig) can be downloaded as a tarball and viewed using a web browser such as Firefox in linux. PyLag's documentation contains numerous examples of running *PyLag*, which have been included to help users get up and running. In order to run the examples, a number of example [input data files](https://drive.google.com/open?id=15UX7Y9JnuLpnPAz700mzmzd917nTClxR) have been made available for download. Further information on their use can be found in the documentation. Last, please note that PyLag has yet to be officially released, and parts of the code and documentation are still under development. This page will be updated when future releases are made.
