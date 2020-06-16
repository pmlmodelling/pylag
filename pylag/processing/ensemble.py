from __future__ import division, print_function

import numpy as np
from scipy import stats
from warnings import warn

from pylag.processing.ncview import Viewer

have_pyqt_fit = True
try:
    from pyqt_fit import kde, kde_methods, kernels
except ImportError:
    have_pyqt_fit = False


def get_probability_density_1D(file_names, dates, depths, depth_bnds, pylag_time_rounding):
    """Compute the ensemble mean concentration in 1D
    
    Particle concentrations are computed on the dates and at the depth levels
    given in the arrays `dates' and `depth'. Each member of the ensemble is
    a separate realisation, with particles starting at the sames locations
    and at the same time in each run. A different method should be used to
    compute probability densities for ensembles in which particles are released
    at different times.
    
    To compute particle concentrations a gaussian kernel density estimator
    is used. Boundaries are treated as being reflecting, thus there is no loss
    of density.
    
    Parameters:
    -----------
    file_names : list[str]
        List of sorted PyLag output files. Each output file corresponds to one member
        of the ensemble.
    
    dates : 1D NumPy array ([t], datetime)
        Dates on which to compute the ensemble mean concentration.
    
    depths : 2D Numpy array ([t, z], float)
        Depths at which to compute the ensemble mean concentration. The array is
        2D, since it may be desirable to have the depths at which concentrations
        are calculated vary in time (e.g. if the model has a moving free
        surface). NB dates.shape[0] must equal depths.shape[0].
    
    depths_bnds : 2D Numpy array ([t, 2], float)
        These are the lower and upper depth bands which are required by the
        kernel method.

    pylag_time_rounding : int
        The number of seconds PyLag outputs should be rounded to.

    Returns:
    --------
    conc : 2D Numpy array (float)
        The concentration at the specified times and depths
    """
    # Function requires pyqt_fit. First check that it is installed
    if not have_pyqt_fit:
        raise RuntimeError("PyQt-fit was not found within this python distribution. Please see PyLag's documentation "
                           "for more information.")

    if dates.shape[0] != depths.shape[0]:
        raise ValueError('Array lengths do not match')
    
    # Array sizes
    n_trials = len(file_names)
    n_times = dates.shape[0]
    n_zlevs = depths.shape[1]
    
    # Use kernel method to estimate density
    dens = np.empty((n_trials, n_times, n_zlevs), dtype=float)
    for i, file_name in enumerate(file_names):
        viewer = Viewer(file_name, time_rounding=pylag_time_rounding)

        # Establish the indices of the time points we want to work with
        time_indices = [viewer.date.tolist().index(date) for date in dates]      
        
        for j, t_idx in enumerate(time_indices):
            zmin = depth_bnds[j, 0]
            zmax = depth_bnds[j, 1]
            est = kde.KDE1D(viewer('z')[t_idx, :].squeeze(), lower=zmin, upper=zmax,
                    method=kde_methods.reflection, kernel=kernels.normal_kernel1d())
            dens[i, j, :] = est(depths[j, :])
    
    return np.mean(dens, axis=0)


def get_probability_density_2D(file_names, time_deltas, x_points, y_points, pylag_time_rounding, group_id=None):
    """Compute the probability density in 2D
    
    Probability densities are computed at the x and y points provided, and at time
    time points after the start of the simulation, as listed in the `time_deltas'
    array. Each member of the ensemble is a separate realisation. Typically, particles
    will have been released from the same points in space in each simulation. Particles
    may or may not have been released at different times.
    
    To compute the probability density a gaussian kernel density estimator is used.
    
    Parameters:
    -----------
    file_names : list[str]
        List of sorted PyLag output files. Each output file corresponds to one member
        of the ensemble.
    
    time_deltas : 1D NumPy array, timedelta
        Time deltas corresponding to times after the start of each ensemble
        simulation at which the probability density should be computed. For
        example, if time_deltas = [1 (day), 2 (days)], then the probability
        density will be computed one and two days after the start of each
        simulation. NB - it is important that data has been saved at
        the specified time point. If it hasn't, an index error will be
        thrown. For example, if the data has been saved at 15 minute time
        intervals, and time_delta = 30 minutes, there should be no problems.
        However, if time_delta = 40 minutes, which cannot be divided exactly by
        15, an error will be generated.
    
    x_points : 1D NumPy array, float
        x points at which to compute the ensemble mean concentration.

    y_points : 1D NumPy array, float
        y points at which to compute the ensemble mean concentration.

    pylag_time_rounding : int
        The number of seconds PyLag outputs should be rounded to.

    group_id : int, optional
        Calculate the probability density for a single group.

    Returns:
    --------
    dens: 2D Numpy array (float)
        The concentration at the specified times and x/y positions.
    """
    
    if len(x_points.shape) != 1 or len(y_points.shape) != 1:
        raise ValueError('x and y input arrays should be 1D')
    
    if x_points.shape[0] != y_points.shape[0]:
        raise ValueError('x and y input arrays should be the same length')
    
    # Array sizes
    n_trials = len(file_names)
    n_times = time_deltas.shape[0]
    n_points = x_points.shape[0]

    # Positions at which to estimate the particle density
    positions = np.vstack([x_points, y_points])

    # Use a kernel method to estimate the density
    dens = np.zeros((n_trials, n_times, n_points), dtype=float)
    for i, file_name in enumerate(file_names):
        viewer = Viewer(file_name, time_rounding=pylag_time_rounding)

        ref_date = viewer.date[0]
        dates = [ref_date + time_delta for time_delta in time_deltas]
        time_indices = [viewer.date.tolist().index(date) for date in dates]

        if group_id is not None:
            group_indices = np.where(viewer('group_id')[:] == group_id)[0]

            for j, t_idx in enumerate(time_indices):
                xpos = viewer('xpos')[t_idx, group_indices].squeeze()
                ypos = viewer('ypos')[t_idx, group_indices].squeeze()
                values = np.vstack([xpos, ypos])
                kernel = stats.gaussian_kde(values)
                dens[i, j, :] = kernel(positions)
        else:
            for j, t_idx in enumerate(time_indices):
                xpos = viewer('xpos')[t_idx, :].squeeze()
                ypos = viewer('ypos')[t_idx, :].squeeze()
                values = np.vstack([xpos, ypos])
                kernel = stats.gaussian_kde(values)
                dens[i, j, :] = kernel(positions)
    return np.mean(dens, axis=0)
