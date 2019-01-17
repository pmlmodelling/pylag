Thu 16 Jul 09:34:35 BST 2015

A Lagrangian modelling framework written in a mixture of Python and Cython.

Installation
------------

0. Register for access to the PML GitLab at http://pml.ac.uk/Modelling_at_PML/Access_Code.
1. git clone git@gitlab.ecosystem-modelling.pml.ac.uk:PyLag/PyLag
2. cd PyLag
3. pip install --user .

Notes
=====

- Currently there appears to be an incompatibility with Cython versions > 0.25.

- There is no current support for compiling PyLag with Intel compilers (GCC works OK).

- The C compiler must support the C++11 standard (i.e. GCC versions >= 4.8, Intel versions >= 12).

