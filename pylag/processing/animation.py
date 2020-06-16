from __future__ import division, print_function

import os
import numpy as np
from netCDF4 import Dataset, num2date
from matplotlib import pyplot as plt

from pylag.processing.utils import round_time
from pylag.processing.utils import get_time_index

from pylag.processing.plot import PyLagPlotter

from PyFVCOM.grid import elems2nodes, find_nearest_point
from PyFVCOM.coordinate import utm_from_lonlat


class Animation(object):
    """Base class for animation objects.
    
    """

    def make_image(self, n):
        pass

    def make_animation(self, tidx_start, tidx_stop, tidx_step=1,
                       fig_dir_name='./figs', verbose=False):
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
        self.plotter = FVCOMPlotter(self.grid_metrics, **kwargs)

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

class ParticleDistributionAnimation(Animation):
    """Produce an animation of the particle probability density field

    Parameters
    ----------
    output_filename: string
        List of netcdf filenames.

    grid_metric_filename: str
        Name of netCDF grid metrics file used for initialising the plotter.

    """

    def __init__(self, output_filenames, grid_metrics_filename, **kwargs):
        # List of output filenames to process
        self.output_filenames = output_filenames

        # The grid metrics file
        self.grid_metrics = Dataset(grid_metrics_filename)

        # Create plotter
        self.plotter = FVCOMPlotter(self.grid_metrics, **kwargs)

        self._process_grid()

    def make_image(self, frame_idx, time_idx):
        # Find the number of particles in each element at time n across all
        # members of the ensemble
        particle_count = np.zeros(self.nele, dtype=float)
        for filename in self.output_filenames:
            d = Dataset(filename,'r')
            hosts = d.variables['host'][time_idx,:]
            for host in hosts:
                particle_count[host] += 1.0

        print('Total particle count is {}'.format(np.sum(particle_count)))

        # Gives depth integrated probability density
        prob_density = particle_count / ( np.sum(particle_count) * self.element_areas )
        prob_density = elems2nodes(prob_density, self.nv, self.nnode)

        # Update plot
        self.plotter.plot_field(prob_density)

        # Update title using advection time
        d = Dataset(self.output_filenames[0],'r')
        date_now = num2date(d.variables['time'][time_idx],units=self.time_units)
        t_advect = (date_now - self.date_start)
        title = "t_advect = {} days {} hours".format(t_advect.days, t_advect.seconds//3600)
        self.plotter.set_title(title)

        return

    def _process_grid(self):
        self.nele = len(self.grid_metrics.dimensions['nele'])
        self.nnode = len(self.grid_metrics.dimensions['node'])
        self.nv = self.grid_metrics.variables['nv'][:].transpose()

        x = self.grid_metrics.variables['x'][:]
        y = self.grid_metrics.variables['y'][:]

        self.element_areas = np.empty(self.nele, dtype=float)
        for idx, nodes in enumerate(self.nv):
            v1 = (x[nodes[0]], y[nodes[0]])
            v2 = (x[nodes[1]], y[nodes[1]])
            v3 = (x[nodes[2]], y[nodes[2]])
            self.element_areas[idx] = self._get_area(v1,v2,v3)

        # Save time units from the first output file
        d = Dataset(self.output_filenames[0],'r')
        time = d.variables['time']
        self.time_units = time.units
        self.date_start = num2date(time[0], units=self.time_units)

    def _get_area(self,v1,v2,v3):
        """
        Return the area of a triangle given the x/y coordinates of its three vertices.

        Parameters:
        -----------
        v1,v2,v2: tuple, (float, float)
            x/y coordinates for the three vertices of a triangle in the x-y plane.

        Returns:
        --------
        area: float
            Area of the triangle.
        """
        area = 0.5 * ( v1[0] * (v2[1] - v3[1]) + v2[0] * ( v3[1] - v1[1] ) + v3[0] * ( v1[1] - v2[1] ) )
        return abs(area) 

class FVCOMSurfaceWindsAnimation(Animation):
    def __init__(self, output_filename, grid_metrics_filename, 
            fvcom_output_period, **kwargs):
        """Produce an animation of FVCOM surface winds
                
        Parameters
        ----------
        output_filename: string
            Name of the netcdf output file.

        grid_metric_filename: str
            Name of netCDF grid metrics file used for initialising the plotter.
            
        bathy : bool
            If true, include a background plot of the bathymetry.

            Default `False'.
        """
        self.fvcom_output_period = fvcom_output_period

        # Plot bathymetry by default
        self.bathy = kwargs.pop('bathy', True)

        # Plot grid by default
        self.overlay_grid = kwargs.pop('overlay_grid', True)

        # Scale for quiver plot
        self.scale = kwargs.pop('scale', 20.0)

        # The grid metrics file
        self.grid_metrics = Dataset(grid_metrics_filename)

        # Create plotter
        self.plotter = FVCOMPlotter(self.grid_metrics, **kwargs)

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
        uwind = self.ds.variables['uwind_speed'][time_idx, :].squeeze()
        vwind = self.ds.variables['vwind_speed'][time_idx, :].squeeze()
        self.plotter.plot_quiver(uwind, vwind, scale=self.scale)

        # Update title with date string for this time index
        time = self.ds.variables['Itime'][time_idx] + self.ds.variables['Itime2'][time_idx] / 1000. / 60. / 60. / 24.
        date = num2date(time, units=self.ds.variables['Itime'].units)
        self.plotter.set_title(date)

        return

class FVCOMSurfaceSalinityAnimation(Animation):
    def __init__(self, output_filename, grid_metrics_filename, 
            fvcom_output_period, **kwargs):
        """Produce an animation of FVCOM surface salinity

        Parameters
        ----------
        output_filename: string
            Name of the netcdf output file.

        grid_metric_filename: str
            Name of netCDF grid metrics file used for initialising the plotter.

        """
        self.fvcom_output_period = fvcom_output_period

        # The grid metrics file
        self.grid_metrics = Dataset(grid_metrics_filename)

        # Create plotter
        self.plotter = FVCOMPlotter(self.grid_metrics, **kwargs)

        # Dataset holding particle positions
        self.ds = Dataset(output_filename)

    def make_image(self, frame_idx, time_idx):
        salinity = self.ds.variables['salinity'][time_idx, 0, :].squeeze()
        self.plotter.plot_field(salinity)

        # Update title with date string for this time index
        time = self.ds.variables['Itime'][time_idx] + self.ds.variables['Itime2'][time_idx] / 1000. / 60. / 60. / 24.
        date = num2date(time, units=self.ds.variables['Itime'].units)
        self.plotter.set_title(date)

        return

class MultiVarAnimation(Animation):
    """Produce an animation of particle positions and FVCOM field vars

    Class to manage the creation of animations that show:

    1) Particle positions
    2) Wind speed and direction
    3) Sea surface elevation
    4) Surface current speed and direction
    5) Bottom current speed and direction
    6) Sea surface salinity

    """
    def __init__(self, pylag_output_filename, fvcom_output_filename,
        grid_metrics_filename, lat_tidal_elev, lon_tidal_elev,
        fvcom_output_period, **kwargs):
        """
                
        Parameters
        ----------
        pylag_output_filename : str
            Name of the pylag output file containing particle data.
            
        fvcom_output_filename : str
            Name of the fvcom output file containing fvcom data.
            
        grid_metrics_filename : str
            Name of the fvcom grid metrics file.
        """
        # Lat and lon coordinates for the tidal elevation subplot
        self.lat_tidal_elev = lat_tidal_elev
        self.lon_tidal_elev = lon_tidal_elev

        # FVCOM output period
        self.fvcom_output_period = fvcom_output_period

        # Plot bathymetry by default
        self.bathy = kwargs.pop('bathy', True)

        # Plot bathymetry by default
        self.overlay_grid = kwargs.pop('overlay_grid', True)

        # List of particle groups to plot
        self.particle_group_ids = kwargs.pop('group_ids', None)

        # List of colours to use for each group
        self.particle_group_colours = kwargs.pop('group_colours', None)

        # PyLag start and end time indices
        self.pylag_tidx_start = kwargs.pop('pylag_tidx_start', 0)
        self.pylag_tidx_end = kwargs.pop('pylag_tidx_end', -1)

        # Stations
        self.stations = kwargs.pop('stations', None)

        # Print updates
        self.verbose = kwargs.pop('verbose', True)

        # Font size for plotting
        try:
            self.fontsize = kwargs['fs']
        except KeyError:
            self.fontsize = 10

        # The grid metrics dataset
        self.grid_metrics = Dataset(grid_metrics_filename)

        # Dataset holding particle positions
        self.pylag_ds = Dataset(pylag_output_filename)

        # Dataset holding fvcom field data
        self.fvcom_ds = Dataset(fvcom_output_filename)

        # Initialise figure
        figsize = kwargs.pop('figsize', (24.,16.))
        self.figure, self.axarr = plt.subplots(2,3,figsize=figsize)
        self.figure.set_facecolor('white')

        # Compute time and date variables for the two data files
        pylag_times = self.pylag_ds.variables['time']
        self.pylag_dates = num2date(pylag_times[:], units=pylag_times.units, calendar=pylag_times.calendar)
        fvcom_times = self.fvcom_ds.variables['Itime'][:] + self.fvcom_ds.variables['Itime2'][:] / 1000. / 60. / 60. / 24.
        date = num2date(time, units=self.ds.variables['Itime'].units)
        fvcom_times = self.fvcom_ds.variables['time']
        fvcom_dates = num2date(fvcom_times[:], units=fvcom_times.units)
        self.fvcom_dates = round_time(fvcom_dates, self.fvcom_output_period)

        # Compute start and end FVCOM time indices based on the PyLag start and end time indices that have been provided
        self.fvcom_tidx_start = get_time_index(self.fvcom_dates, self.pylag_dates[self.pylag_tidx_start])
        self.fvcom_tidx_end = get_time_index(self.fvcom_dates, self.pylag_dates[self.pylag_tidx_end])

        # Node index for the tidal elevation plot
        x, y, z = utm_from_lonlat([self.lon_tidal_elev], [self.lat_tidal_elev])
        _, _, _, nodes = find_nearest_point(self.fvcom_ds.variables['x'], self.fvcom_ds.variables['y'], x[0], y[0])
        self.tidal_elev_node = nodes[0]
        if self.verbose:
            print('Plotting tidal elevations at point [{}E, {}N]'.format(
                   self.fvcom_ds.variables['lon'][self.tidal_elev_node],
                   self.fvcom_ds.variables['lat'][self.tidal_elev_node]))

        # Create a dictionary of plotting objects
        self.plotters = {}

        # Tidal elevations - does not require a FVCOMPlotter object
        # ---------------------------------------------------------
        zeta = self.fvcom_ds.variables['zeta'][self.fvcom_tidx_start:self.fvcom_tidx_end,self.tidal_elev_node]
        dates = self.fvcom_dates[self.fvcom_tidx_start:self.fvcom_tidx_end]
        self.axarr[0,0].plot(dates, zeta)
        self.axarr[0,0].set_xlabel('Time', fontsize=self.fontsize)
        self.axarr[0,0].set_ylabel('Elevation (m)', fontsize=self.fontsize)
        self.axarr[0,0].set_title('Sea surface elevation [{}, {}]'.format(
                self.fvcom_ds.variables['lon'][self.tidal_elev_node],
                self.fvcom_ds.variables['lat'][self.tidal_elev_node]),
                fontsize=self.fontsize)
        self.axarr[0,0].set_xlim(dates[0], dates[-1])
        self.figure.autofmt_xdate()

        # Salinity plotter
        # ----------------
        self.plotters['salinity'] = FVCOMPlotter(self.grid_metrics, figure=self.figure,
                axes=self.axarr[1,0], title='Surface salinity', vmin=30.0, vmax=35.0, cmap='seismic',
                **kwargs)

        # Create particle positions plotter
        self.plotters['particle_positions_plotter'] = FVCOMPlotter(self.grid_metrics, figure=self.figure, 
                axes=self.axarr[0,1], stations=self.stations, title='Particle positions', vmin=-150.0,
                vmax=0.0, cmap='gray', **kwargs)

        # Create surface wind speed plotter
        self.plotters['surface_winds_plotter'] = FVCOMPlotter(self.grid_metrics, figure=self.figure, 
                axes=self.axarr[1,1], title='Surface wind velocity',  vmin=-150.0,
                vmax=0.0, cmap='gray', **kwargs)

        # Create surface current plotter
        self.plotters['surface_currents_plotter'] = FVCOMPlotter(self.grid_metrics, figure=self.figure,
                axes=self.axarr[0,2], title='Surface current velocity',  vmin=-150.0,
                vmax=0.0, cmap='gray', **kwargs)

        # Create bottom current plotter
        self.plotters['bottom_currents_plotter'] = FVCOMPlotter(self.grid_metrics, figure=self.figure, 
                axes=self.axarr[1,2], title='Bottom current velocity',   vmin=-150.0,
                vmax=0.0, cmap='gray', **kwargs)

        # Add bathymetry from the grid metrics file?
        if self.bathy is True:
            h = -self.grid_metrics.variables['h'][:]
            for plotter in self.plotters.values():
                plotter.plot_field(h)

        # Add bathymetry from the grid metrics file?
        if self.overlay_grid is True:
            for plotter in self.plotters.values():
                plotter.overlay_grid()

    def make_image(self, frame_idx, time_idx):
        # Compute t indices for FVCOM
        n_fvcom = get_time_index(self.fvcom_dates, self.pylag_dates[time_idx])
        print('PyLag time is {}'.format(self.pylag_dates[time_idx]))
        print('FVCOM time is {}'.format(self.fvcom_dates[n_fvcom]))

        # Plot current tidal elevation
        # ----------------------------
        self.axarr[0,0].scatter(self.fvcom_dates[n_fvcom], self.fvcom_ds.variables['zeta'][n_fvcom, self.tidal_elev_node],
                s=80, color='r', edgecolors='none')

        # Plot surface salinity
        # ---------------------
        salinity = self.fvcom_ds.variables['salinity'][n_fvcom, 0, :].squeeze()
        self.plotters['salinity'].plot_field(salinity)

        # Plot particle positions
        # -----------------------
        x = self.pylag_ds.variables['xpos']
        y = self.pylag_ds.variables['ypos']
        gids = self.pylag_ds.variables['group_id']

        if (self.particle_group_ids is not None) and (self.particle_group_colours is not None): 
            for group_id, group_colour in zip(self.particle_group_ids, self.particle_group_colours):
                indices = np.where(gids[:] == group_id)[0]
                self.plotters['particle_positions_plotter'].plot_scatter(x[time_idx,indices].squeeze(),
                        y[time_idx,indices].squeeze(), group_id, group_colour)
        else:
            self.plotters['particle_positions_plotter'].plot_scatter(x[time_idx,:].squeeze(), y[time_idx,:].squeeze())

        # Plot surface wind speed
        # -----------------------
        uwind = self.fvcom_ds.variables['uwind_speed'][n_fvcom, :].squeeze()
        vwind = self.fvcom_ds.variables['vwind_speed'][n_fvcom, :].squeeze()
        self.plotters['surface_winds_plotter'].plot_quiver(uwind, vwind, scale=20)

        # Plot surface currents
        # ---------------------
        u = self.fvcom_ds.variables['u'][n_fvcom, 0, :].squeeze()
        v = self.fvcom_ds.variables['v'][n_fvcom, 0, :].squeeze()
        self.plotters['surface_currents_plotter'].plot_quiver(u, v, scale=1.0)

        # Plot bottom currents
        # ---------------------
        u = self.fvcom_ds.variables['u'][n_fvcom, -1, :].squeeze()
        v = self.fvcom_ds.variables['v'][n_fvcom, -1, :].squeeze()
        self.plotters['bottom_currents_plotter'].plot_quiver(u, v, scale=0.1)

        # Add annotation giving the current date and time
        if hasattr(self, 'annotation'):
            self.annotation.set_text('{}'.format(self.pylag_dates[time_idx]))
        else:
            self.annotation = self.axarr[0,0].annotate('{}'.format(self.pylag_dates[time_idx]),
                    xy=(.475, .975), xycoords='figure fraction', horizontalalignment='left',
                    verticalalignment='top', fontsize=self.fontsize)

        return
