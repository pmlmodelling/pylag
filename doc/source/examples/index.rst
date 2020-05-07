.. _examples:

********
Examples
********

PyLag can be fed inputs from a number of different sources, including both analytical and numerical models of
different flow fields, or direct observations (e.g. from HF Radar facilities). Inputs may be read in from disk or
calculated on the fly. Below a number of example use cases of are described. These start with very simple analytical
models of different flows, the results of which are used to verify the model implementation, and to test the accuracy of
different numerical integration schemes. More involved examples include using PyLag with the General Ocean Turbulence
Model (GOTM) and the Finite Volume Community Ocean Model (FVCOM). In both cases, the results of the particle tracking
model are compared with the those of a eulerian model of tracer dispersion. In the final example, PyLag is used with
FVCOM to investigate connectivity along the SWUK coastline.


.. toctree::
    :maxdepth: 2

    lateral_adv_analytic
    lateral_adv_diff_analytic
    fvcom_forward_tracking

