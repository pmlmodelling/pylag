from __future__ import division, print_function

import os
import numpy as np
from netCDF4 import Dataset, num2date
from matplotlib import pyplot as plt

from pylag.processing.utils import round_time
from pylag.processing.utils import get_time_index

from pylag.processing.plot import PyLagPlotter


class Animation(object):
    """Base class for animation objects.
    
    """

    def make_image(self, n):
        pass

    def make_animation(self, tidx_start, tidx_stop, tidx_step=1,
                       fig_dirname='./figs', verbose=False):
        """ Create PyLag animation and save to file.
        
        Parameters
        ----------
        tidx_start : int
            Starting time index.
        
        tidx_stop : int
            Finishing time index.
        
        tidx_step : int
            Time index step size.
        
        fig_dirname : str
            Name of the directory in which to save individual figures.
            
            Default `./figs'.

        verbose : bool
            Print progress information.
            
            Default `False'.
        """
        # Create figure directory if it does not exist already
        if not os.path.isdir('{}'.format(fig_dirname)):
            os.mkdir('{}'.format(fig_dirname))

        # Generate all image files
        for i, tidx in enumerate(range(tidx_start, tidx_stop, tidx_step)):
            if verbose: print("Making image for time index {}.".format(tidx))
            self.make_image(i,tidx)

        plt.savefig('{}/image_{:03d}.png'.format(fig_dirname, frame_idx), dpi=300)


class ParticleTrajectoryAnimation(Animation):
    def __init__(self, output_filename, grid_metrics_filename, **kwargs):
        """Produce an animation of particle movement over time.
                
        Parameters
        ----------
        output_filename: string
            Name of the netcdf output file.

        grid_metric_filename: str
            Name of netCDF grid metrics file used for initialising the plotter.

        path_lines : bool
            If true, plot particle path lines in addition to the particle's
            current location.
            
            Default `False'.
            
        bathy : bool
            If true, include a background plot of the bathymetry.

            Default `False'.

        group_ids : int
            LIst of group IDs to plot.
            
            Default `None'.

        group_colours : str
            List of colours to use when plotting.

            Default `None'.

        TDDO
        ----
        Following recent updates to the code, this won't work. Needs updating to support the latest API in plot.py.
        """
        # Plot bathymetry by default
        self.bathy = kwargs.pop('bathy', True)

        # Plot grid by default
        self.overlay_grid = kwargs.pop('overlay_grid', True)
        
        # Plot path lines by default
        self.path_lines = kwargs.pop('path_lines', True)

        # List of particle groups to plot
        self.group_ids = kwargs.pop('group_ids', None)

        # List of colours to use for each group
        self.group_colours = kwargs.pop('group_colours', None)

        # The grid metrics file
        self.grid_metrics = Dataset(grid_metrics_filename)

        # Create plotter
        self.plotter = PyLagPlotter(self.grid_metrics, **kwargs)

        # Dataset holding particle positions
        self.ds = Dataset(output_filename)
        
        # Add bathymetry from the grid metrics file?
        if self.bathy is True:
            h = -self.grid_metrics.variables['h'][:]
            self.plotter.plot_field(h)

        # Overlay the grid
        if self.overlay_grid is True:
            self.plotter.draw_grid()
        
    def make_image(self, frame_idx, time_idx):
        x = self.ds.variables['xpos']
        y = self.ds.variables['ypos']
        gids = self.ds.variables['group_id']

        if (self.group_ids is not None) and (self.group_colours is not None): 
            for group_id, group_colour in zip(self.group_ids, self.group_colours):
                indices = np.where(gids[:] == group_id)[0]
                self.plotter.plot_scatter(x[time_idx,indices].squeeze(),
                        y[time_idx,indices].squeeze(), group_name=group_id, colour=group_colour)

                # Add path lines?
                if self.path_lines is True:
                    self.plotter.plot_lines(x[:time_idx,indices], y[:time_idx,indices],
                            group_name=group_id, colour=group_colour)
        else:
            self.plotter.plot_scatter(x[time_idx,:].squeeze(), y[time_idx,:].squeeze())

            # Add path lines?
            if self.path_lines is True:
                self.plotter.plot_lines(x[:time_idx+1,:], y[:time_idx+1,:])

        # Update title with date string for this time index
        time_units = self.ds.variables['time'].units
        date = num2date(self.ds.variables['time'][time_idx], units=time_units)
        self.plotter.set_title(date)
        
        return
