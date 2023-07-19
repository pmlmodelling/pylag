# PyLag

[![Build and test](https://github.com/pmlmodelling/pylag/actions/workflows/pylag-package-conda-linux.yml/badge.svg)](https://github.com/pmlmodelling/pylag/actions)
[![Documentation Status](https://readthedocs.org/projects/pylag/badge/?version=latest)](https://pylag.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/jimclark/pylag/badges/version.svg)](https://anaconda.org/jimclark/pylag)
[![Anaconda-Server Badge](https://anaconda.org/jimclark/pylag/badges/downloads.svg)](https://anaconda.org/jimclark/pylag)


PyLag is an offline particle tracking model. The model expects as inputs time independent and/or dependent variables that describe the state of a given fluid. These may be measured quantities or the predictions of an analytical or numerical model. Using these, the model computes Lagrangian trajectories for particles released into the fluid at a particular point in time and space. The model is primarily aimed at marine applications, but is intended to be versatile enough to be extended to other contexts; for example, studies of atmospheric dispersion.

PyLag was created with the aim to make available a particle tracking model that is a) fast to run, b) easy to use, c) extensible and d) flexible. The model is written in a mixture of [Python](http://www.python.org) and [Cython](http://www.cython.org).

## How to cite

Uncles, R. J., Clark, J. R., Bedington, M., Torres, R. 2020. “On sediment dispersal in the Whitsand Bay Marine Conservation Zone: Neighbor to a closed dredge-spoil disposal site” in Marine Protected Areas: Evidence, Policy and Practise, ed Robert Clark and John Humphreys (Elsevier Inc.).

## Getting started

Full installations instructions and tutorial examples can be found in [PyLag's documentation](https://pylag.readthedocs.io/en/latest/).

