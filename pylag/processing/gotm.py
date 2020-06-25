"""
Tools to assist with analysing GOTM based outputs
"""

from __future__ import division, print_function

import numpy as np
from warnings import warn

from pylag.processing.ncview import Viewer

have_pyqt_fit = True
try:
    from pyqt_fit import kde, kde_methods, kernels
except (ImportError, ModuleNotFoundError):
    have_pyqt_fit = False


def get_rmse(gotm_file_name, gotm_var_name, pylag_file_names, dates, gotm_time_rounding,
             pylag_time_rounding, mass_factor=1.0):
    """Compute the RMSE

    Compute the RMSE between GOTM and PyLag model outputs.

    Parameters
    ----------
    gotm_file_name : str
        Name of the file containing output from the Eulerian model. At the
        moment, only GOTM-based outptus are supported.

    gotm_var_name : str
        The name of the GOTM variable we will comare the output of the particle
        tracking model with.

    pylag_file_names : list[str]
        List of sorted PyLag output files. Each output file corresponds to one member
        of the ensemble.

    dates : 1D NumPy array ([t], datetime)
        Dates on which to compute the emsemble mean concentration.

    gotm_time_rounding : int
        The number of seconds GOTM outputs should be rounded to.

    pylag_time_rounding : int
        The number of seconds PyLag outputs should be rounded to.

    Returns
    -------
    rmse : float
        The RMSE.
    """
    # Function requires pyqt_fit. First check that it is installed
    if not have_pyqt_fit:
        raise RuntimeError("PyQt-fit was not found within this python distribution. Please see PyLag's documentation "
                           "for more information.")

    # Create GOTM viewer
    gotm_viewer = Viewer(gotm_file_name, time_rounding=gotm_time_rounding)

    # Find GOTM date indices
    gotm_date_indices = [gotm_viewer.date.tolist().index(date) for date in dates]

    # Array sizes
    n_trials = len(pylag_file_names)
    n_times = dates.shape[0]
    n_zlevs = gotm_viewer('z').shape[1]

    diff_squared = np.empty((n_trials, n_times, n_zlevs), dtype=float)
    for i, pylag_file_name in enumerate(pylag_file_names):
        pylag_viewer = Viewer(pylag_file_name, time_rounding=pylag_time_rounding)

        # Find PyLag date indices
        pylag_date_indices = [pylag_viewer.date.tolist().index(date) for date in dates]

        # Compare the two models
        for j, (gotm_idx, pylag_idx) in enumerate(zip(gotm_date_indices, pylag_date_indices)):
            # Extract the GOTM variable
            gotm_conc = gotm_viewer(gotm_var_name)[gotm_idx, :].squeeze()

            # Compute z limits
            zmin = gotm_viewer('zi')[gotm_idx, 0].squeeze()
            zmax = gotm_viewer('zi')[gotm_idx, -1].squeeze()

            # Extract depths to be used for the comparison
            depths = gotm_viewer('z')[gotm_idx, :].squeeze()

            # Compute the particle concentration
            est = kde.KDE1D(pylag_viewer('z')[pylag_idx, :].squeeze(),
                            lower=zmin, upper=zmax, method=kde_methods.reflection,
                            kernel=kernels.normal_kernel1d())
            pylag_conc = est(depths) * mass_factor

            # Compute the squared difference
            diff_squared[i, j, :] = (pylag_conc[:] - gotm_conc[:]) ** 2.0

    return np.sqrt(np.mean(diff_squared))