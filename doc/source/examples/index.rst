.. _examples:

********
Examples
********

PyLag can be fed inputs from a number of different sources, including both analytical and numerical models of
different flow fields, or direct observations (e.g. from HF Radar facilities). Inputs may be read in from disk or
calculated on the fly. Below a number of example use cases are described. These start with very simple analytical
models of different flows, the results of which are used to verify the model implementation, and to test the accuracy of
different numerical integration schemes. More involved examples include using PyLag directly with different ocean models,
or with inputs defined on common structured and unstructured grid types. Where possible, these try to highlight different
model features, such as running forward or backward in time simulations; running with Cartesian or Polar Coordinates;
performing ensemble simulations; and exploiting PyLag's parallel computing support.


.. toctree::
    :maxdepth: 2

    lateral_adv_analytic
    lateral_adv_diff_analytic
    vertical_mixing_analytic
    vertical_mixing_numeric
    fvcom_forward_tracking
    fvcom_backward_tracking
    arakawa_a_forward_tracking
    roms_txla

