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

PyLag is tailored to work on computers running Linux. A small number of
libraries and packages must be installed before PyLag itself can be installed 
and run. These include the following system libraries and packages:

* Python 2.7 - The python 2.7 interpreter
* GSL - The GNU Scientific Library for numerical analysis
* GSL-devel - Libraries and header files for GSL development
* Cython - A language for writing python extension modules
* CythonGSL - Cython wrappers for GSL

and the Python packages:

* NumPy - Numerical Python for fast array operations
* NetCDF4-python - Python/NumPy interface to netCDF
* Matplotlib - Python plotting library
* ProgressBar - Progress bar for tracking run progress

Many of these may be already installed on your system. If not, you will need
to install these using your package manager. For example, running the following
command on Fedora with root privliges will install GSL and GSL-devel::

    $ sudo yum install GSL GSL-devel


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

Once you have a obtained a copy of the code, simply run the setup.py scripts for
the two projects in order to install them. For example, to perform a local 
installation given the above directory structure run::

    $ cd $HOME/code/git/PyLag/PyLag
    $ python setup.py install --user
    $ cd $HOME/code/git/PyLag/PyLag-tools
    $ python setup.py install --user
    
You can check that PyLag has been successfully installed by running the
commands::

    $ python -c "import pylag"
    $ python -c "import pylagtools"

which should exit without error.
