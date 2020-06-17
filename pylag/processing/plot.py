from __future__ import division, print_function

import numpy as np
from scipy import interp
import collections
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
from cftime import num2pydate
from cmocean import cm
from warnings import warn

from PyFVCOM.coordinate import lonlat_from_utm
from PyFVCOM.grid import nodes2elems
from PyFVCOM.plot import cm2inch

from pylag.processing.ncview import Viewer
from pylag.processing.utils import round_time
from pylag.processing.ensemble import get_probability_density_1D

have_basemap = True
try:
    from mpl_toolkits.basemap import Basemap
except (ImportError, ModuleNotFoundError):
    have_basemap = False


class PyLagPlotter:
    """Create PyLag plot objects
    
    Class to assist in the creation of plots and animations. This is the
    default PyLag plotter, designed to work with PyLag simulation output
    that has been generated using input data that is defined on a single
    horizontal mesh. The mesh is read from the run's grid metrics file,
    which must be passed to PyLagPlotter during class initialisation.

    Specifically, PyLagPlotter will work with:

    1) Arakawa A-grid derived data
    2) FVCOM derived data
    """
    def __init__(self, grid_metrics_file, axes, fs=10,
                 use_basemap=True, extents=None, res='c', tick_inc=None,
                 fill_continents=False, draw_coastlines=False):
        """
        Parameters:
        -----------
        grid_metrics_file : Dataset or str
            This is either the path to a NetCDF FVCOM grid metrics file, or a
            NetCDF Dataset. If the former, FVCOMPlotter will try to instantiate
            a new Dataset using the supplied file name.

        axes : Axes, optional
            Maplotplotlib Axes object.

        fs : int, optional
            Font size to use when rendering plot text

        use_basemap : boolean, optional
            Boolean specifying whether or not to use basemap to create a 2D map
            on top of which the data will be plotted. The default option is
            `True'. If `False', a simple Cartesian grid is drawn instead.

        extents : 1D array, optional
            Four element numpy array giving lon/lat limits (e.g. [-4.56, -3.76,
            49.96, 50.44])
        
        res : string, optional
            Resolution to use when drawing Basemap object

        tick_inc : list, optional
            Add coordinate axes (i.e. lat/long) at the intervals specified in
            the list ([lon_spacing, lat_spacing]).

        fill_continents : boolean, optional
            Colour continents
        
        draw_coastlines : boolean, optional
            Draw coastlines

        Author(s):
        -------
        James Clark (PML)
        Pierre Cazenave (PML)

        """
        if isinstance(grid_metrics_file, Dataset):
            ds = grid_metrics_file
        elif isinstance(grid_metrics_file, str):
            ds = Dataset(grid_metrics_file, 'r')
        else:
            raise ValueError("`grid_metrics_file' should be either a "\
                    "pre-constructed netCDF.Dataset or a srting giving the "\
                    "path to the file.")

        self.axes = axes
        self.figure = axes.get_figure()
        self.fs = fs
        self.use_basemap = use_basemap
        self.extents = extents
        self.res = res
        self.tick_inc = tick_inc
        self.draw_coastlines = draw_coastlines
        self.fill_continents = fill_continents

        # If the user intends to use Basemap, confirm that it is installed.
        if not have_basemap and self.use_basemap:
            raise RuntimeError('Basemap was not found within this python distribution. To generate non-basemap '
                               'plots using FVCOMPlotter set use_basemap = False. See PyLag-s documentation '
                               'for more information.')

        # Initialise the figure
        self.__init_figure(ds)

        # Close the NetCDF file for reading
        ds.close()
        del ds

    def __init_figure(self, ds):
        # Read in the required grid variables
        self.n_nodes = len(ds.dimensions['node'])
        self.n_elems = len(ds.dimensions['element'])
        self.nv = ds.variables['nv'][:]

        # Try to read the element mask
        try:
            self.maskc = ds.variables['mask'][:]
        except KeyError:
            self.maskc = None

        if self.use_basemap:
            self.lon = ds.variables['longitude'][:]
            self.lat = ds.variables['latitude'][:]
            self.lonc = ds.variables['longitude_c'][:]
            self.latc = ds.variables['latitude_c'][:]
        else:
            self.x = ds.variables['x'][:]
            self.y = ds.variables['y'][:]
            self.xc = ds.variables['xc'][:]
            self.yc = ds.variables['yc'][:]

        # Triangles
        self.triangles = self.nv.transpose()

        # Create basemap object?
        if self.use_basemap:
            # If plot extents were not given, use min/max lon/lat values
            if self.extents is None:
                self.extents = np.array([self.lon.min(),
                                         self.lon.max(),
                                         self.lat.min(),
                                         self.lat.max()])

            self.m = Basemap(llcrnrlon=self.extents[0:2].min(),
                             llcrnrlat=self.extents[-2:].min(),
                             urcrnrlon=self.extents[0:2].max(),
                             urcrnrlat=self.extents[-2:].max(),
                             rsphere=(6378137.00,6356752.3142),
                             resolution=self.res,
                             projection='merc',
                             area_thresh=0.1,
                             lat_0=self.extents[-2:].mean(),
                             lon_0=self.extents[0:2].mean(),
                             lat_ts=self.extents[-2:].mean(),
                             ax=self.axes)

            self.m.drawmapboundary()

            if self.fill_continents:
                self.m.fillcontinents(zorder=2, linewidth=0.2)

            if self.draw_coastlines:
                self.m.drawcoastlines(zorder=3, linewidth=0.2)

            if self.tick_inc:
                meridians = np.rint(np.arange(np.min(self.extents[:2]), np.max(self.extents[:2]), self.tick_inc[0]))
                parallels = np.rint(np.arange(np.min(self.extents[2:]), np.max(self.extents[2:]), self.tick_inc[1]))
                self.m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=self.fs, linewidth=0.1, ax=self.axes)
                self.m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=self.fs, linewidth=0.1, ax=self.axes)

            # Save x and y coordinates for plotting - these are calculated
            # internally by the Basemap class instance
            self._x, self._y = self.m(self.lon, self.lat)
            self._xc, self._yc = self.m(self.lonc, self.latc)
        else:
            # If plot extents were not given, use min/max x/y values
            if self.extents is None: self.extents = np.array([self.x.min(), 
                                                              self.x.max(), 
                                                              self.y.min(), 
                                                              self.y.max()])
            self.axes.set_xlim(self.extents[0], self.extents[1])
            self.axes.set_ylim(self.extents[2], self.extents[3])

            # Here, x and y coordinates are simply the cartesian coordinates
            self._x = self.x
            self._y = self.y
            self._xc = self.xc
            self._yc = self.yc

        # Store two triangulations - one masked and one not
        self.tri = Triangulation(self._x, self._y, self.triangles, mask=self.maskc)

    def plot_field(self, field, add_colorbar=True, cb_label=None, cb_ticks=None, **kwargs):
        """Map the given FVCOM field.
        
        Parameters:
        -----------
        field : 1D NumPy array
            Field to plot.
        """
        # Update array values if the plot has already been initialised
        if hasattr(self, 'tripcolor_plot'):
            self.tripcolor_plot.set_array(field)
            return

        # Create tripcolor plot
        self.tripcolor_plot = self.axes.tripcolor(self.tri, field, **kwargs)

        # Add colobar scaled to axis width
        if add_colorbar:
            cax = create_cbar_ax(self.axes)
            self.cbar = self.figure.colorbar(self.tripcolor_plot, cax=cax, ticks=cb_ticks)
            self.cbar.ax.tick_params(labelsize=self.fs)
            if cb_label:
                self.cbar.set_label(cb_label, fontsize=self.fs)

        return

    def plot_lines(self, x, y, group_name='Default', zone='30N', coordinate_system='cartesian', **kwargs):
        """Plot path lines.

        In addition to the listed parameters, the function accepts all keyword arguments taken by the Matplotlib
        plot command.

        Parameters:
        -----------
        x : 1D array TOCHECK
            Array of x coordinates to plot.

        y : 1D array TOCHECK
            Array of y coordinates to plot.
        
        group_name : str, optional
            Group name for this set of particles - a separate plot object is
            created for each group name passed in.
            
            Default `None'
            
        zone : string, optional
            See PyFVCOM documentation for a full list of supported codes.

        """
        alpha = kwargs.pop('alpha', 0.25)
        color = kwargs.pop('color', 'r')
        linewidth = kwargs.pop('linewidth', 1.0)

        if not hasattr(self, 'line_plot'):
            self.line_plot = {}
        
        # Remove current line plots for this group, if they exist
        if group_name in self.line_plot:
            if self.line_plot[group_name]:
                self.remove_line_plots(group_name)

        if self.use_basemap:
            if coordinate_system == 'cartesian':
                lon, lat = lonlat_from_utm(x, y, zone)
            elif coordinate_system == 'spherical':
                lon, lat = x, y

            mx, my = self.m(lon, lat)
        else:
            mx = x
            my = y

        self.line_plot[group_name] = self.axes.plot(mx, my, zorder=3, alpha=alpha, color=color, linewidth=linewidth,
                                                    **kwargs)

    def remove_line_plots(self, group_name):
        """Remove line plots for group `group_name'
        
        Parameters:
        -----------
        group_name : str
            Name of the group for which line plots should be deleted.

        """
        if hasattr(self, 'line_plot'):
            while self.line_plot[group_name]:
                self.line_plot[group_name].pop(0).remove()

    def plot_scatter(self, x, y, zone='30N', group_name='Default', coordinate_system='cartesian', **kwargs):
        """Plot scatter.

        Any extra keyword arguments are passed to Matplotlib's scatter function.

        Parameters:
        -----------
        x : 1D array
            Array of x coordinates to plot.

        y : 1D array
            Array of y coordinates to plot.
        
        group_name : str, optional
            Group name for this set of particles - a separate plot object is
            created for each group name passed in.
            
            Default `None'

        coordinate_system : str, optional
            The coordinate system used for 'x' and 'y'. Options are:
                - cartesian (x and y values in m)
                - spherical (x an y values are lons and lats in deg)

            Default: cartesian

        zone : string, optional
            See PyFVCOM documentation for a full list of supported codes.

            Default `30N'

        """
        if not hasattr(self, 'scat_plot'):
            self.scat_plot = dict()
        
        if self.use_basemap:
            if coordinate_system == 'cartesian':
                lon, lat = lonlat_from_utm(x, y, zone)
            elif coordinate_system == 'spherical':
                lon, lat = x, y

            mx, my = self.m(lon, lat)
        else:
            mx = x
            my = y

        try:
            data = np.array([mx, my])
            self.scat_plot[group_name].set_offsets(data.transpose())
        except KeyError:
            self.scat_plot[group_name] = self.axes.scatter(mx, my, zorder=4, **kwargs)

    def mark_location(self, x, y, ** kwargs):
        """ Mark location

        Mark the given locations on the map. This is very similar to plot_scatter, but
        no attempt is made to update existing plots.

        Parameters:
        -----------
        x : 1D array
            Array of x coordinates to plot.

        y : 1D array
            Array of y coordinates to plot.
        """
        if self.use_basemap:
            mx, my = self.m(x, y)
        else:
            mx = x
            my = y

        self.axes.scatter(mx, my, zorder=5, **kwargs)

    def draw_grid(self, draw_masked_elements=False, **kwargs):
        """ Draw the underlying grid or mesh

        Parameters:
        -----------
        draw_masked_elements : bool
            Include masked elements. Default False.
        """
        reinstate_mask = False
        if self.maskc is not None and draw_masked_elements:
            reinstate_mask = True
            self.tri.set_mask(None)

        self.axes.triplot(self.tri, zorder=2, **kwargs)

        # Reinstate the mask if needed
        if reinstate_mask:
            self.tri.set_mask(self.maskc)

    def set_title(self, title):
        self.axes.set_title(title, fontsize=self.fs)

    def get_nodal_coords(self):
        return np.copy(self.x), np.copy(self.y)


class GOTMPlotter(object):
    """Class to assist in the creation of GOTM plot objects

    Class to assist in the creation of plots and animations based on output
    from the GOTM model, including additional support to plot PyLag outputs.

    """

    def __init__(self, file_name, fs=10, time_rounding=None):
        """
        GOTMPlotter presently supports the creation of the following plot types:

        "time_series" : Plot variable through time at a given depth

        "profile" : Plot depth profile

        "hovmoller" : pcolormesh plot of a variable on a depth - time grid

        "hovmoller_particles" : pcolormesh plot of particle concentrations on a depth - time grid

        "scatter" : scatter plot of particle positions on a depth - time grid

        "pathlines" : line plot of particle pathlines on a depth - time grid

        See function documentation for more details.

        Parameters:
        -----------
        file_name : str
            File from which to read grid info.

        fs : int, optional
            Font size to use when rendering plot text

        time_rounding : int
            Period between saved data points (in seconds) which is used
            to round datetime objects.

        Author(s):
        -------
        James Clark (PML)

        """
        self.file_name = file_name
        self.fs = fs
        self.time_rounding = time_rounding

        # Initialise the figure
        self.__init_figure()

    def __init_figure(self):
        # Initialise dataset
        self.ds = Dataset(self.file_name, 'r')

        # Times/dates
        self.times = self.ds.variables['time']
        self.dates = num2pydate(self.times[:], units=self.times.units,
                                calendar=self.times.calendar)

        # Time and date bands (for plotting with pcolormesh)
        dt = self.times[1] - self.times[0]
        self.time_bnds = np.empty(self.times.shape[0] + 1, dtype=self.times.dtype)
        self.time_bnds[:-1] = self.times[:] - dt / 2
        self.time_bnds[-1] = self.times[-1] + dt / 2
        self.date_bnds = num2pydate(self.time_bnds[:], units=self.times.units,
                                  calendar=self.times.calendar)

        # Round dates
        if self.time_rounding:
            self.dates = round_time(self.dates, self.time_rounding)
            self.date_bnds = round_time(self.date_bnds, self.time_rounding)

        # Depth at layer centres
        self.z = self.ds.variables['z'][:].squeeze()

        # Depth as layer interfaces
        self.zi = self.ds.variables['zi'][:].squeeze()

        # Layer separations
        self.h = self.ds.variables['h'][:].squeeze()

        # Construct depth and time grids for use with pcolormesh; coordinates
        # should correspond to the points of quadrilaterals surrounding the
        # points where variables are defined. NB the position of the
        # quadrilaterals changes depending on whether the variable is defined at
        # layer interfaces (e.g. turbulence vars) or layer centres (e.g.
        # passive tracers)

        # Compute z bands for plotting with pcolormesh
        self.z_bnds = np.empty((self.z.shape[0] + 1, self.z.shape[1] + 1), dtype=float)
        for i in range(self.z_bnds.shape[1]):
            self.z_bnds[:, i] = interp(self.time_bnds[:], self.times[:], self.zi[:, i])

        # Compute zi bands for plotting with pcolormesh:
        # a) First compute zi_bnds based on the depth of cell centres. Layer
        # thicknesses are used to extrapolate beyond the edge of the grid.
        # b) Next, interpolate these values to time points that are offset by
        # dt/2,
        zi_bnds = np.empty((self.zi.shape[0], self.zi.shape[1] + 1), dtype=float)
        zi_bnds[:, 0] = self.z[:, 0] - self.h[:, 0]
        zi_bnds[:, 1:-1] = self.z[:, :]
        zi_bnds[:, -1] = self.z[:, -1] + self.h[:, -1]
        self.zi_bnds = np.empty((self.zi.shape[0] + 1, self.zi.shape[1] + 1), dtype=float)
        for i in range(self.zi_bnds.shape[1]):
            self.zi_bnds[:, i] = interp(self.time_bnds[:], self.times[:], zi_bnds[:, i])

        # Compute date bands for use with both z_bnds and zi_bnds
        self.date_z_bnds = np.tile(self.date_bnds[:], [self.z_bnds.shape[1], 1]).T
        self.date_zi_bnds = np.tile(self.date_bnds[:], [self.zi_bnds.shape[1], 1]).T

    def time_series(self, axes, var_name, depth, **kwargs):
        """ Make a time series plot

        The function plots a time series of the given variable at the given depth below the free surface.
        GOTM variable data is first interpolated to the given depth.

        Parameters:
        -----------
        axes : matplotlib.axes.Axes
            Axes object

        var_name : str
            Name of variable to plot.

        depth : float
            Depth relative to the free surface (= 0 m). Positive up.

        Returns:
        --------
        axes : matplotlib.axes.Axes
            Axes object
        """
        # Variable data
        var = self.ds.variables[var_name]

        # Interpolate variable data to the given depth below the moving free surface
        var_time_series = []
        for i in range(var.shape[0]):
            depth_offset = depth + self.zi[i, -1]  # Remove offset introduced by the moving free surface

            var_time_series.append(interp(depth_offset, self.z[i, :], var[i, :].squeeze()))

        axes.plot(self.dates, var_time_series, **kwargs)
        axes.set_xlabel('Time', fontsize=self.fs)
        axes.set_ylabel('{} ({})'.format(var_name, var.units), fontsize=self.fs)

        return axes

    def profile(self, axes, var_name, date):
        """ Generate a depth profile of the listed variable at the given time point

        Parameters:
        -----------
        axes : matplotlib.axes.Axes
            Axes object.

        var_name : str
            The variable to plot.

        date : datetime
            The date on which to extract the profile.


        Returns:
        --------
        axes : Matplotlib.axes.Axes
            Axes object.
        """
        # Calculate the model time index using a nearest neighbour approach
        t_idx = (np.abs(np.array(self.dates) - date)).argmin()

        var = self.ds.variables[var_name]

        axes.plot(var[t_idx, :, 0, 0].squeeze(), self.z[t_idx, :])

        # Add axis labels
        axes.set_xlabel('{} ({})'.format(var_name, var.units), fontsize=self.fs)
        axes.set_ylabel('Depth (m)', fontsize=self.fs)

        return axes

    def hovmoller(self, axes, var_name, add_colorbar=True, cb_label=None, cb_ticks=None, **kwargs):
        """ Draw a hovmoller diagram

        Parameters:
        -----------
        axes : matplotlib.axes.Axes
            Axes object

        var_name : str
            Name of variable to plot.

        cb_label : str, optional
            The colour bar label.

        cb_ticks : list[float], optional
            Colorbar ticks.

        Returns:
        --------
        axes : matplotlib.axes.Axes
            Axes object
        """

        var = self.ds.variables[var_name]

        # Is the variable defined at layer centers or layer interfaces?
        if 'z' in var.dimensions:
            depth_grid = self.z_bnds
            time_grid = self.date_z_bnds
        elif 'zi' in var.dimensions:
            depth_grid = self.zi_bnds
            time_grid = self.date_zi_bnds
        else:
            raise ValueError("Variable `{}' is not depth resolved".format(var_name))

        plot = axes.pcolormesh(time_grid, depth_grid, var[:].squeeze(), **kwargs)

        # Set depth lims
        axes.set_ylim([depth_grid.min(), depth_grid.max()])

        # Add axis labels
        axes.set_xlabel('Time', fontsize=self.fs)
        axes.set_ylabel('Depth (m)', fontsize=self.fs)

        # Add colour bar
        if add_colorbar:
            figure = axes.get_figure()
            self.add_colour_bar(figure, axes, plot, cb_label, cb_ticks)

        return axes

    def hovmoller_particles(self, axes, file_names, ds, de, time_rounding, mass_factor=1.0, add_colorbar=True,
                            cb_label=None, cb_ticks=None, **kwargs):
        """ Plot particle concentrations

        Parameters:
        -----------
        axes : matplotlib.axes.Axes
            Axes object

        file_names : list[str]
            List of sorted PyLag output files. Each output file corresponds to one member
            of the ensemble.

        ds : datetime
            Start datetime.

        de : datetime
            End datetime.

        time_rounding : int
            Period between saved data points (in seconds) which is used to round
            PyLag datetime objects. This option is included to account for cases
            in which PyLag times are written to file with limited precision. Once
            rounded, two datetime objects can be more easily compared. Note this
            parameter may be different to the GOTM time_rounding parameter, which
            is an instance variable.

        mass_factor : float
            Multiplier that is used to generate concentrations from
            probability densities.

        add_colorbar : bool, optional
            Add colorbar?

        cb_label : bool, optional
            Colorbar label.

        cb_ticks : list[float], optional
            Colorbar ticks.
        """
        pylag_viewer = Viewer(file_names[0], time_rounding=time_rounding)

        pylag_first_idx = pylag_viewer.date.tolist().index(ds)
        pylag_last_idx = pylag_viewer.date.tolist().index(de)
        pylag_dates = pylag_viewer.date[pylag_first_idx:pylag_last_idx + 1]

        gotm_first_idx = self.dates.tolist().index(ds)
        gotm_last_idx = self.dates.tolist().index(de)
        gotm_dates = self.dates[gotm_first_idx:gotm_last_idx + 1]

        if not np.array_equal(pylag_dates, gotm_dates):
            raise RuntimeError('PyLag and GOTM date arrays do not match.')

        # Compute particle concentrations
        depths = self.z[gotm_first_idx:gotm_last_idx + 1, :].squeeze()
        depth_bnds = self.zi[gotm_first_idx:gotm_last_idx + 1, (0, -1)].squeeze()
        conc = get_probability_density_1D(file_names, pylag_dates, depths, depth_bnds, time_rounding) * mass_factor

        # Compute date and depth bands for plotting with pcolormesh. The +2
        # accounts for 1) Pyhton slicing rules, and 2) the fact pcolormesh wants
        # the date and z band arrays to be one bigger in size than the
        # concentration array.
        pcol_date_bnds = self.date_z_bnds[gotm_first_idx:gotm_last_idx + 2, :]
        pcol_depth_bnds = self.z_bnds[gotm_first_idx:gotm_last_idx + 2, :]

        # Plot
        plot = axes.pcolormesh(pcol_date_bnds, pcol_depth_bnds, conc, **kwargs)

        # Set depth lims
        axes.set_ylim([pcol_depth_bnds.min(), pcol_depth_bnds.max()])

        # Add axis labels
        axes.set_xlabel('Time', fontsize=self.fs)
        axes.set_ylabel('Depth (m)', fontsize=self.fs)

        # Add colour bar
        if add_colorbar:
            figure = axes.get_figure()
            self.add_colour_bar(figure, axes, plot, cb_label, cb_ticks)

        return axes

    def plot_scatter(self, axes, dates, zpos, **kwargs):
        """ Scatter plot of particle positions through time

        """
        for i in range(zpos.shape[1]):
            axes.scatter(dates, zpos[:, i], **kwargs)

        # Set time and depth lims
        axes.set_xlim([np.min(dates), np.max(dates)])
        axes.set_ylim([np.min(zpos), np.max(zpos)])

    def plot_pathlines(self, axes, dates, zpos, **kwargs):
        """ Plot particle pathlines through time

        """
        axes.plot(dates, zpos[:, :], **kwargs)

        # Set time and depth lims
        axes.set_xlim([np.min(dates), np.max(dates)])
        axes.set_ylim([np.min(zpos), np.max(zpos)])

        return axes

    def add_colour_bar(self, figure, axes, plot, cb_label, cb_ticks):
        # Add colour bar scaled to axis width
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = figure.colorbar(plot, cax=cax, ticks=cb_ticks)
        cbar.ax.tick_params(labelsize=self.fs)
        cbar.set_label(cb_label)

    def set_title(self, title):
        self.axes.set_title(title, fontsize=self.fs)


def plot_particle_positions(output_filename, grid_metrics_filename, time_index, **kwargs):
    """Plot the position of particles.
    
    Plot particle positions. If path_lines keyword is supplied, include a plot
    of path lines.
    
    Parameters
    ----------
    output_filename : List, string
        PyLag output files to use when plotting particle positions.
    
    grid_metrics_filename : str
        Name of file providing grid info.
    
    time_index : int
        Time index at which to plot data.
    """

    # Plot bathymetry by default
    bathy = kwargs.pop('bathy', True)

    # Plot grid by default
    overlay_grid = kwargs.pop('overlay_grid', True)

    # Plot path lines by default
    path_lines = kwargs.pop('path_lines', True)

    # List of particle groups to plot
    group_ids = kwargs.pop('group_ids', None)

    # List of colours to use for each group
    group_colours = kwargs.pop('group_colours', None)
    
    # Grid metrics file
    grid_metrics = Dataset(grid_metrics_filename)

    # Create plotter
    plotter = PyLagPlotter(grid_metrics, **kwargs)

    # Dataset holding particle positions
    ds = Dataset(output_filename, 'r')

    # Add bathymetry from the grid metrics file?
    if bathy is True:
        h = -grid_metrics.variables['h'][:]
        plotter.plot_field(h)

    # Overlay the grid
    if overlay_grid is True:
        plotter.draw_grid()

    x = ds.variables['x']
    y = ds.variables['y']
    gids = ds.variables['group_id']

    if (group_ids is not None) and (group_colours is not None): 
        for group_id, group_colour in zip(group_ids, group_colours):
            indices = np.where(gids[:] == group_id)[0]
            plotter.plot_scatter(x[time_index,indices].squeeze(),
                    y[time_index, indices].squeeze(), group_id, group_colour)

            # Add path lines?
            if path_lines is True:
                plotter.plot_lines(x[:time_index+1, indices], y[:time_index+1, indices],
                        group_id, group_colour)
    else:
        plotter.plot_scatter(x[time_index, :].squeeze(), y[time_index, :].squeeze())

        # Add path lines?
        if path_lines is True:
            plotter.plot_lines(x[:time_index, :], y[:time_index, :])

    # Update title with date string for this time index
    time_units = ds.variables['time'].units
    date = num2pydate(ds.variables['time'][time_index], units=time_units)
    plotter.set_title(date)


def create_figure(figure_size=(10., 10.),  font_size=10, axis_position=None, projection=None, bg_color='white'):
    """ Create a Figure object

    Parameters:
    -----------

    figure_size : tuple(float), optional
        Figure size in cm. This is only used if a new Figure object is
        created.

    font_size : int
        Font size to use for axis labels

    axis_position : 1D array, optional
        Array giving axis dimensions

    bg_color : str, optional
        Colour to use for the axis background. Default is `white'. When
        creating a figure for plotting FVCOM outputs, it can be useful
        to set this to `gray'. When FVCOM is fitted to a coastline, the
        gray areas mark the land boundary used by the model. This provides
        a fast alternative to plotting a high resolution (e.g. `res' = 'f')
        land boundary using methods provided by the Basemap class instance.

    Author(s):
    -------
    James Clark (PML)

    """
    figure_size_inches = (cm2inch(figure_size[0]), cm2inch(figure_size[1]))
    figure = plt.figure(figsize=figure_size_inches)
    figure.set_facecolor('white')

    axes = figure.add_subplot(1, 1, 1, projection=projection)

    if axis_position:
        axes.set_position(axis_position)

    axes.tick_params(axis='both', which='major', labelsize=font_size)
    axes.tick_params(axis='both', which='minor', labelsize=font_size)

    try:
        axes.set_facecolor(bg_color)
    except AttributeError:
        axes.set_axis_bgcolor(bg_color)

    return figure, axes


def create_cbar_ax(ax):
    """Create colorbar axis alligned with plot axis y limits

    Parameters
    ----------
    ax : Axes
        Plot axes instsance

    Returns
    -------
    cax : Axes
        Colorbar plot axis
    """
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="5%", pad=0.05)


def colourmap(variable):
    """ Use a predefined colour map for a given variable.

    Leverages the cmocean package for perceptually uniform colour maps.

    Parameters
    ----------
    variable : str, iterable
        For the given variable name(s), return the appropriate colour palette from the cmocean/matplotlib colour maps.
        If the variable is not in the pre-defined variables here, the returned values will be 'viridis'.

    Returns
    -------
    colourmaps : matplotlib.colours.cmap, dict
        The colour map(s) for the variable(s) given.

    """

    default_cmap = mpl.cm.get_cmap('viridis')

    cmaps = {'q2': cm.dense,
             'l': cm.dense,
             'q2l': cm.dense,
             'tke': cm.dense,
             'viscofh': cm.dense,
             'kh': cm.dense,
             'nuh': cm.dense,
             'teps': cm.dense,
             'tauc': cm.dense,
             'temp': cm.thermal,
             'sst': cm.thermal,
             'salinity': cm.haline,
             'zeta': cm.balance,
             'ww': cm.balance,
             'omega': cm.balance,
             'uv': cm.speed,
             'uava': cm.speed,
             'speed': cm.speed,
             'u': cm.delta,
             'v': cm.delta,
             'ua': cm.delta,
             'va': cm.delta,
             'uvanomaly': cm.delta,
             'direction': cm.phase,
             'uvdir': cm.phase,
             'N1_p': cm.dense,
             'N3_n': cm.dense,
             'N4_n': cm.dense,
             'N5_s': cm.dense,
             'O2_o': cm.oxy,
             'O3_c': cm.matter,
             'O3_TA': cm.dense,
             'R1_c': cm.matter,
             'R2_c': cm.matter,
             'R4_c': cm.turbid,
             'R6_c': cm.turbid,
             'R8_c': cm.turbid,
             'P1_c': cm.algae,
             'P2_c': cm.algae,
             'P3_c': cm.algae,
             'P4_c': cm.algae,
             'P1_Chl': cm.algae,
             'P2_Chl': cm.algae,
             'P3_Chl': cm.algae,
             'P4_Chl': cm.algae,
             'Z4_c': cm.algae,
             'Z5_c': cm.algae,
             'Z6_c': cm.algae,
             'light_EIR': cm.solar,
             'light_parEIR': cm.solar,
             'light_xEPS': cm.solar,
             'light_iopABS': cm.solar,
             'O2_eO2mO2': cm.oxy,
             'O2_osat': cm.oxy,
             'O2_AOU': cm.oxy,
             'O3_pH': cm.ice,
             'O3_pCO2': cm.dense,
             'O2_fair': cm.curl,
             'O3_fair': cm.curl,
             'O3_wind': cm.curl,
             'h_morpho': cm.deep,
             'h': cm.deep,
             'h_r': cm.deep_r,
             'bathymetry': cm.deep,
             'bathymetry_r': cm.deep_r,
             'taub_total': cm.thermal,
             'mud_1': cm.turbid,
             'mud_2': cm.turbid,
             'sand_1': cm.turbid,
             'sand_2': cm.turbid,
             'todal_ssc': cm.turbid,
             'total_ssc': cm.turbid,
             'mud_1_bedfrac': cm.dense,
             'mud_2_bedfrac': cm.dense,
             'sand_1_bedfrac': cm.dense,
             'sand_2_bedfrac': cm.dense,
             'mud_1_bedload': cm.dense,
             'mud_2_bedload': cm.dense,
             'sand_1_bedload': cm.dense,
             'sand_2_bedload': cm.dense,
             'bed_thick': cm.deep,
             'bed_age': cm.tempo,
             'bed_por': cm.turbid,
             'bed_diff': cm.haline,
             'bed_btcr': cm.thermal,
             'bot_sd50': cm.turbid,
             'bot_dens': cm.thermal,
             'bot_wsed': cm.turbid,
             'bot_nthck': cm.matter,
             'bot_lthck': cm.matter,
             'bot_dthck': cm.matter,
             'bot_morph': cm.deep,
             'bot_tauc': cm.thermal,
             'bot_rlen': cm.dense,
             'bot_rhgt': cm.dense,
             'bot_bwav': cm.turbid,
             'bot_zdef': cm.dense,
             'bot_zapp': cm.dense,
             'bot_zNik': cm.dense,
             'bot_zbio': cm.dense,
             'bot_zbfm': cm.dense,
             'bot_zbld': cm.dense,
             'bot_zwbl': cm.dense,
             'bot_actv': cm.deep,
             'bot_shgt': cm.deep_r,
             'bot_maxD': cm.deep,
             'bot_dnet': cm.matter,
             'bot_doff': cm.thermal,
             'bot_dslp': cm.amp,
             'bot_dtim': cm.haline,
             'bot_dbmx': cm.dense,
             'bot_dbmm': cm.dense,
             'bot_dbzs': cm.dense,
             'bot_dbzm': cm.dense,
             'bot_dbzp': cm.dense,
             'wet_nodes': cm.amp,
             'tracer1_c': cm.dense,
             'DYE': cm.dense}

    if isinstance(variable, collections.Iterable) and not isinstance(variable, str):
        colourmaps = []
        for var in variable:
            if var in cmaps:
                colourmaps.append(cmaps[var])
            else:
                colourmaps.append(default_cmap)
        # If we got a list of a single value, return the value rather than a list.
        if len(colourmaps) == 1:
            colourmaps = colourmaps[0]
    else:
        if variable in cmaps:
            colourmaps = cmaps[variable]
        else:
            colourmaps = default_cmap

    return colourmaps


