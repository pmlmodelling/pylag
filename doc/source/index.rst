PyLag
=====

PyLag is an offline particle tracking model. The model expects as inputs time independent and/or dependent variables that describe the state of a given fluid. These may be measured quantities or the predictions of an analytical or numerical model. Using these, the model computes Lagrangian trajectories for particles released into the fluid at a particular point in time and space. The model is primarily aimed at marine applications, but is versatile enough to be used in other contexts too; for example, in studies of atmospheric dispersion.

PyLag was created with the aim to make available a particle tracking model that is a) fast to run, b) easy to use, c) extensible and d) flexible. The model is written in a mixture of `Python <http://www.python.org>`_ and `Cython <http://www.cython.org>`_,

How to cite
===========

Uncles, R. J., Clark, J. R., Bedington, M., Torres, R. 2020. “On sediment dispersal in the
Whitsand Bay Marine Conservation Zone: Neighbor to a closed dredge-spoil disposal site” in
Marine Protected Areas: Evidence, Policy and Practise, ed Robert Clark and John Humphreys
(Elsevier Inc.).

Contents
========

.. toctree::
    :maxdepth: 2

    install/installation
    examples/index
    documentation/index
    licensing
    acknowledgements

