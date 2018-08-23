PyLag
=====

PyLag is an offline particle tracking model. The model expects as inputs time independent and/or dependent variables that describe the state of a given fluid. These may be measured quantities or the predictions of an analytical or numerical model. Using these, the model computes Lagrangian trajectories for particles released into the fluid at a particular point in time and space. The model is primarily aimed at marine applications, but is versatile enough to be used in other contexts too; for example, in studies of atmospheric dispersion.

PyLag was created with the aim to make available a particle tracking model that is a) fast to run, b) easy to use, c) extensible and d) flexible. The model is written in a mixture of `Python <http://www.python.org>`_ and `Cython <http://www.cython.org>`_,


Supported models
================

At present, PyLag includes in built support for input data from the following sources:

*  `General Ocean Turbulence Model (GOTM) <http://gotm.net/>`_
*  `Finite Volume Community Ocean Model (FVCOM) <http://fvcom.smast.umassd.edu/fvcom/>`_


Contents
========

.. toctree::
    :maxdepth: 2

    install/installation
    examples/index
    documentation/index
    acknowledgements/acknowledgements

