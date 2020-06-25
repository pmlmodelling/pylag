"""
Tools to assist with analysing FVCOM based outputs
"""
from __future__ import division, print_function

import numpy as np
from scipy import stats

from PyFVCOM.read import FileReader as FVCOMFileReader

from pylag.processing.ncview import Viewer


def get_rmse(fvcom_file_name, fvcom_var_name, pylag_file_names, dates, pylag_time_rounding, mass_factor=1.0,
             n_points=None):
    """Compute the RMSE

    Compute the RMSE between FVCOM and PyLag model outputs.

    Parameters
    ----------
    fvcom_file_name : str
        Name of the file containing output from the Eulerian model. At the
        moment, only FVCOM-based outptus are supported.

    fvcom_var_name : str
        The name of the FVCOM variable we will comare the output of the particle
        tracking model with.

    pylag_file_names : list[str]
        List of sorted PyLag output files. Each output file corresponds to one member
        of the ensemble.

    dates : 1D NumPy array ([t], datetime)
        Dates on which to compute the emsemble mean concentration.

    pylag_time_rounding : int
        The number of seconds PyLag outputs should be rounded to.

    mass_factor : float
        Factor used to convert probability densities into concentrations.

    n_points : int, optional
        Restrict the comparison to the `n` points which have the highest
        FVCOM tracer concentration. If None, all grid points are compared.

    Returns
    -------
    rmse : float
        The RMSE.

    """
    # Create FVCOM File Reader
    fvcom_reader = FVCOMFileReader(fvcom_file_name)

    # Find FVCOM date indices
    fvcom_date_indices = get_fvcom_date_indices(fvcom_reader, dates)

    # Array sizes
    n_trials = len(pylag_file_names)
    n_times = dates.shape[0]
    if n_points is None:
        n_points = fvcom_reader.grid.x.shape[0]

    # Compute the squared difference
    diff_squared = np.empty((n_trials, n_times, n_points), dtype=float)
    for i, pylag_file_name in enumerate(pylag_file_names):
        viewer = Viewer(pylag_file_name, time_rounding=pylag_time_rounding)

        # Find PyLag date indices
        pylag_date_indices = [viewer.date.tolist().index(date) for date in dates]

        # Compare the two models
        for j, (fvcom_idx, pylag_idx) in enumerate(zip(fvcom_date_indices, pylag_date_indices)):
            fvcom_conc = get_fvcom_var(fvcom_reader, fvcom_var_name, fvcom_idx)

            # Restrict the comparison to the nodes with the highest concentrations by first sorting the fvcom_conc
            # array, then slicing it to get the indices corresponding to the nodes with the highest concentrations
            indices = np.argsort(fvcom_conc)[-n_points:]

            # Positions at which to estimate the particle density
            positions = np.vstack([fvcom_reader.grid.x[indices], fvcom_reader.grid.y[indices]])

            # Compute the concentration from particle positions
            xpos = viewer('x')[pylag_idx, :].squeeze()
            ypos = viewer('y')[pylag_idx, :].squeeze()
            values = np.vstack([xpos, ypos])
            kernel = stats.gaussian_kde(values)
            pylag_conc = kernel(positions) * mass_factor

            # Compute the squared difference
            diff_squared[i, j, :] = (pylag_conc[:] - fvcom_conc[indices]) ** 2.0

    return np.sqrt(np.mean(diff_squared))


def get_fvcom_date_indices(fvcom_reader, dates):
    """ Get FVCOM date indices

    """
    return [fvcom_reader.time.datetime.tolist().index(date) for date in dates]


def get_fvcom_var(fvcom_reader, fvcom_var_name, time_index, depth_integrated=True):
    """ Extract data for the given variable at the given time index

    """
    fvcom_reader.load_data([fvcom_var_name], dims={'time': [time_index]})
    fvcom_tracer = getattr(fvcom_reader.data, fvcom_var_name).squeeze()

    if not depth_integrated:
        return fvcom_tracer

    fvcom_reader.load_data(['zeta'], dims={'time': [time_index]})

    zeta = fvcom_reader.data.zeta.squeeze()
    h = fvcom_reader.grid.h
    siglev = fvcom_reader.grid.siglev

    # For comparison in 2D, compute the depth integral
    dz = np.abs(np.diff(siglev, axis=0)) * (zeta + h)

    return np.sum(dz * fvcom_tracer, axis=0)


__all__ = ["get_rmse"]
