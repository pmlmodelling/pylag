.. _getting_started:

***************
Getting started
***************

.. _registration:

Registration
============

New users must first register to access the code. To do this, navigate to the 
webpage <TODO>, check the PyLag tickbox and fill out the registration form 
before clicking submit. You should then receive an email confirming your 
registration. Each registration request is processed manually, so please be
patient if you don't receive an email with you login details straight away!

.. _requirements:

Requirements
============

PyLag is tailored to work in a UNIX/Linux environment. A small number of
libraries and packages must be installed before PyLag itself can be installed 
and run. These include the following system libraries and packages:

* `Python 2.7 <https://www.python.org/download/releases/2.7>`_ - The python 2.7 interpreter
* `GSL <https://www.gnu.org/software/gsl/>`_ - The GNU Scientific Library for numerical analysis
* `GSL-devel <https://www.gnu.org/software/gsl/>`_ - Libraries and header files for GSL development

and the Cython/Python packages:

* `Cython <http://cython.org/>`_ - A language for writing python extension modules
* `CythonGSL <https://pypi.python.org/pypi/CythonGSL>`_ - Cython wrappers for GSL
* `NumPy <http://www.numpy.org/>`_ - Numerical Python for fast array operations
* `NetCDF4 <http://unidata.github.io/netcdf4-python/>`_ - Python/NumPy interface to netCDF
* `ProgressBar <https://pypi.python.org/pypi/progressbar>`_ - Progress bar for tracking run progress
* `Natsort <https://pypi.python.org/pypi/natsort>`_ - Python sorting algorithm.
* `PyFVCOM <https://pypi.python.org/pypi/PyFVCOM>`_ - FVCOM utilities

These can either be installed using your package manager or, in the case or python
packages, using `pip <https://pip.pypa.io/en/stable/>`_. Different operating 
systems use different package managers, and package names may be slightly 
different. Recent releases of `Fedora <https://getfedora.org/>`_ use the 
`dnf package manager <https://fedoraproject.org/wiki/Dnf>`_ which is the successor
to `yum <https://fedoraproject.org/wiki/Yum>`_. `ubuntu <http://www.ubuntu.com/>`_ uses
the `apt package manager <https://wiki.debian.org/Apt>`_.

Using dnf, GSL and GSL-devel can be installed with the command:

    $ sudo dnf install GSL GSL-devel

Note that you must have administrator privileges to install new software using 
dnf. dnf will automatically work out any dependencies and install these for you.

Before installing PyLag, `Cython <http://cython.org/>`_ and 
`CythonGSL <https://pypi.python.org/pypi/CythonGSL>`_ must be installed. Both can be installed
using `pip <https://pip.pypa.io/en/stable/>`_ If pip is not installed on your system, install
it using your linux package manager (e.g. dnf). With pip installed, run:

    $ pip install --user Cython

    $ pip install --user CythonGSL

pip searches `PyPi <https://pypi.python.org/pypi>`_ for the named packages and installs them
along with any dependencies. The --user flag tells pip to install these packages locally,
meaning administrator privliges are not required.

Many of the python packages required by PyLag can also be installed using your package manager.
Note these tend to be out of date relative to the newest releases. As some of PyLag's dependencies
include binary extensions (e.g. `NumPy <http://www.numpy.org/>`_), it is often faster
to install these using your package manager as they come pre-compiled. If you don't do 
this, pip will attempt to install these package before installing PyLag, although the
installation will take more time.

To build PyLag's documentation several other dependencies must be satisfied:

* `sphinx <http://www.sphinx-doc.org/en/stable/>`_ - Python documentation generator
* `sphinx_rtd_theme <https://pypi.python.org/pypi/sphinx_rtd_theme>`_ - Spinx theme for readthedocs.org

On fedora 23 these can be install using the command::

    $ sudo dnf install python-sphinx python-sphinx_rtd_theme python-sphinx-latex

Here, python-sphinx-latex is a useful package that satisfies many latex dependencies 
required to build a pdf of the documentation.

.. _download:

Download
========

The preferred means of accessing the code is through the source code management
system `git <https://git-scm.com/>`_, which is free open-source software 
available for most common computing platforms. By using git, users can easily 
stay up to date with the latest PyLag release, report issues or track bugfixes. 
For PyLag, git access is enabled through the git management system 
`GitLab <https://gitlab.ecosystem-modelling.pml.ac.uk>`_.

Once you have `registered <registration_>`_ to access the code a GitLab account
will be created for you. If you checked the PyLag tickbox on the registration 
page you will have been automatically added to the 
`PyLag Group <https://gitlab.ecosystem-modelling.pml.ac.uk/groups/PyLag>`_.

The code is distributed in two distinct packages which make up the PyLag Group. 
The first contains the PyLag source code, the second a set of tools
to help with setting up and analysing PyLag simulations. When accessing the code
using git, it is recommended that users use secure shell (SSH) to communicate 
with the GitLab server. This allows users to establish a secure connection 
between their computer and GitLab, and to easily pull and push repositories.
To enable access via SSH, users must first add a public SSH key to GitLab.

To begin, first check for existing SSH keys on your computer::

    $ cd $HOME/.ssh
    $ ls -la

If neither id_rsa.pub or id_dsa.pub are listed, a new SSH key must first be
generated. This can be achieved using the following commands::

    $ ssh-keygen -t rsa -C "username@email.com"
    $ ssh-add $HOME/.ssh/id_rsa

Now add the SSH key to GitLab. To do this, first copy the key to the clipboard::

    $ xclip -sel clip < $HOME/.ssh/id_rsa.pub

Then in a web browser, login to 
`GitLab <https://gitlab.ecosystem-modelling.pml.ac.uk>`_, click on 
*Profile settings* (third icon from the left in the top right hand corner of 
the GitLab webpage), and then the *SSH Keys* tab. Click *Add SSH Key*, then 
enter a title (e.g. *Work PC*), and paste the key from the clipboard into *Key*.
Finally, click *Add key*, which will save the new key.

With SSH access setup, you can now clone the PyLag repository::

    $ mkdir -p $HOME/code/git/PyLag && cd $HOME/code/git/PyLag
    $ git clone https://gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag.git>
    $ git clone https://gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag-tools.git>

If you don't want to use git to access the code, you can always grab a copy by
downloading and unpacking tarballs of the two repositories.


.. _installation:

Installation
============

Once you have a obtained a copy of the code, install
PyLag and PyLag-tools using the python package manager 
`pip <https://pip.pypa.io/en/stable/>`_. For example, to perform a local 
installation given the above directory structure run::

    $ cd $HOME/code/git/PyLag/PyLag
    $ pip install -r requirements.txt
    $ pip install -e .
    $ cd $HOME/code/git/PyLag/PyLag-tools
    $ pip install -e .

pip will automatically search through PyLag's python dependencies and try to install these
if they are not found. All of this will be done locally, meaning root privileges are not required.
    
You can check that PyLag and PyLag-tools have been successfully installed by running the
commands::

    $ python -c "import pylag"
    $ python -c "import pylagtools"

which should exit without error.

