# Release status and notes

PyLag is still very much in development. This page will be updated when future releases are made.

## Version 0.5.2 29/06/2021

* Explicitly declare that plots within the documentation be incorporated inline.

* Fix problem when setting extents in FVCOMPlotter when using cartesian coordinates.

* Fix problem with zeta not being initialised when not supplied as an input variable (specific to Arakawa A grids, and 3D tracking).

* Fix typo in variable library for the name of thetao variable (specific to Arakawa A gridded data).

* Allow for alternative delimiters in FVCOM obc file. 

## Version 0.5.1 18/05/2021

* Fix Kz and Ah variable names in FVCOMDataReader.

## Version 0.5 07/05/2021

* Add ability to interpolate within boundary elements with masked nodes.

* Implement reflecting boundary conditions in geographic coordinates.

* Add multiple code optimisations to reduce run times.

* Add global run tutorial to docs.

* Add support for including a Smagorinksy-type eddy diffusivity term which is computed from the velocity field.

* Switch to using the psi grid to identify boundary crossings when working with ROMS.

## Version 0.4 17/02/2020

* Add basic infrastructure to support individual based modelling.

* Add the ability to simulate particle mortality with accompanying tutorial example.

* Add the ability to specify non-standard dimension and variable names when creating an Arakawa A-grid metric file.

* Fix bug associated with the incorrect flagging of open boundaries by stripy.

* Record the version of PyLag use to create the grid metrics file in the global attributes to assist with version consistency checking.

* Switch to using the land sea element mask for identifying land elements in grids that have masked entries. This yields a significant improvement in speed when creating the particle seed should many of the particle lie outside of the model domain.

* Add quiver plotting tool to assist with plotting the velocity field.

* Implement restoring horizontal boundary condition and clearly distinguish this from reflecting conditions in cartesian and geographic coordinates.

* Add fix for numerical issues associated with the use of acos from the c library.

## Version 0.3.3 23/01/2020

* Switch out stripy for scipy when creating regional triangulations from an Arakawa A-grid in order to fix problems associated with the treatment of open boundaries.

## Version 0.3.2 09/01/2020

* Fix boundary error that occasionally arises with wetting and drying.

## Version 0.3.1 23/12/2020

* Fix negative phi issue with geographic coords on an Arakawa A grid.

## Version 0.3 22/12/2020

* Update installation instructions regarding issue #35

## Version 0.2 11/12/2020

* Add support for irregular time arrays
* Fix bug when trimming polar latitudes
* Reduce the number of essential dependencies

## Version 0.1 - 18/11/2020

* Code made publicly available via [GitHub](https://github.com/jimc101/PyLagGitHub)
