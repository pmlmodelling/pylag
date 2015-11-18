Thu 16 Jul 09:34:35 BST 2015

Lagrangian modelling framework.

...

Suggest we start this by reimplementing the FVCOM particle tracking code in python. First thoughts:

1) We should add support for a configuration parsing tool. Suggest we use ConfigParser, and have it read in a configuration file similar to that used with the FVCOM lag tool. We can then pass a configuration object to any object that requires it (assuming no copy). We could implement hardcoded defaults that are overridden by anything in the configuration file that is in turn overridden by anything passed in as command line arguments.

2) We should create a tool for accessing velocity fields stored in various formats (netcdf, binary etc), and stored on multiple grid types (structured, unstructured etc). Suggest we stub out something equivalent to an abstract base class that defines the interface all derived classes must adhere to.

3) Numerical integration. Hopefully SciPy has something we can use off the shelf, which will likely be faster than anything we write, and will avoid the pain of implementing RK4 etc. I guess we want to support a few different algorithms, set in the configuration file.

4) Velocity interpolation. Again, hopefully we can use some speedy well-tested Numpy or Scipy scheme.

5) Lagrangian metrics. Including FTLE, FSLE, Strain Lines ....

6) Outputs ... suggest we create a class for writing outputs and metrics to NETCDF, along with some standard analysis tools.

More thoughts ...

Useful design patterns we may consider employing are Builder and AbstractFactory. The former might come in useful if we hava a set of objects that must be created together (e.g. objects for processing unstructured vs structured grids), the latter for assembling a collection of objects on the fly given what is specified in a configuration file. See https://github.com/faif/python-patterns for examples in python.

For reading FVCOM output, we can use PySeidon (https://github.com/GrumpyNounours/PySeidon), which already has a neat class structure for reading and interrogating data (including subsetting in time and space).
