"""
Tools to assist with opening and inspecting PyLag input and output files
"""
from __future__ import division, print_function

from netCDF4 import Dataset
from cftime import num2pydate

from pylag.processing.utils import round_time


class Viewer(object):
    """ Class to provide easy access to data stored in netCDF format
    
    This class can be used to view and access data in PyLag data files. In
    general it will work with GOTM outputs too, although FVCOM outputs are
    are best read using PyFVCOM's FileReader class.

    Code adapted from the python module ncdfView.py written by Momme
    Butenschon (CMCC, formerly PML).

    Attributes
    ----------
    _filename : str
        Full or absolute path to the data file
    
    """
    def __init__(self, filename, mask=True, quiet=True, time_rounding=None):
        """Open the file for reading
        
        Parameters
        ----------
        filename : str
            Full or absolute path to the file.
        
        mask : bool, optional
            Apply mask to file variables.
        
        quiet : bool, optional
            Print updates to stdout.

        time_rounding : int
            Period between saved data points (in seconds) which is used
            to round datetime objects. This option is included to account for cases
            in which PyLag times are written to file with limited precision. Once
            rounded, two datetime objects can be more easily compared.
        """
        if not quiet:
            print('Opening netCDF file {} ...'.format(filename))

        self._ds = Dataset(filename, 'r')

        if mask:
            for var in self._ds.variables.values():
                var.set_auto_maskandscale(True)

        self._time_rounding = time_rounding

        self._time = None
        self._date = None

    @property
    def time(self):
        """ Time array
        """
        if self._time is not None:
            return self._time

        for key, var in list(self._ds.variables.items()):
            if key in ['time', 'time_centered', 'time_ref']:
                self._time = var[:]
                return self._time
        raise KeyError('Time variable not found')

    @property
    def date(self):
        """ Date array
        """
        if self._date is not None:
            return self._date

        for key, var in list(self._ds.variables.items()):
            if key in ['time', 'time_centered', 'time_ref']:
                self._date = num2pydate(var[:], units=var.units, calendar=var.calendar)

                if self._time_rounding is not None:
                    self._date = round_time(self._date, self._time_rounding)

                return self._date
        raise KeyError('Time variable not found')
    
    def __call__(self, var_str, Object=True, Squeeze=True):
        if Object:
            return self._ds.variables[var_str]
        else:
            if Squeeze:
                a = self._ds.variables[var_str][:].squeeze().copy()
                return a
            else:
                a = self._ds.variables[var_str][:].copy()
                return a

    def __str__(self):
        info_str='-----------------\n'+\
            'netCDF Object:\n'+\
            '-----------------'
        for key in self._ds.ncattrs():
            info_str += '\n\n'+key+':\t'+str(getattr(self, key))
        dim_list = self._ds.dimensions.items()
        dim_list = [(key, dim, len(dim)) for key, dim in dim_list]
        dim_list.sort(cmp=lambda x, y: cmp(x[0], y[0]))
        for key, dim, size in dim_list:
            info_str += '\n\t'+key
            if dim.isunlimited(): 
                info_str += '\tUNLIMITED => '+str(size)
            else:
                info_str += '\t'+str(size)
        info_str += '\n\n' + 'Variables:\n'
        var_list = self._ds.variables.items()
        var_list.sort(cmp=lambda x, y: cmp(x[0], y[0]))
        for key, var in var_listist:
            info_str += '\n\t' + key + ':'
            for k in var.ncattrs():
                if k == 'long_name':
                    info_str += '\t'+str(getattr(var, k))
                elif k=='units':
                    info_str += '\t'+'['+str(getattr(var, k) + ']')
            info_str += '\n\t\t' + str(var.dimensions) + '=' + str(var.shape)
            info_str += '\t' + str(var.dtype)
        return info_str
 
    def var_info(self, var_str):
        """Print variable info
        
        Parameters
        ----------
        var_str : string
            Variable name/key
        """
        try:
            var = self._ds.variables[var_str]
            info_str = '\n\t' + var_str + str(var.dimensions) + ': ' + str(var.shape)\
                    + '\t' + str(var.dtype)
            for k in var.ncattrs():
                info_str += '\n\t\t' + k + ':\t' + str(getattr(var, k)) + '\n'
            print(info_str)
        except KeyError:
            print('Variable "' + var_str + '" not found!')

    def list_vars(self):
        """List all variables stored in the file"""

        print('File contains the variables ...')
        for key, var in self._ds.variables.items():
            try:
                long_name = var.long_name
            except AttributeError:
                long_name = None
            print('{}: {}'.format(key, long_name))

    def close(self):
        """Close dataset and exit"""
        self._ds.close()
