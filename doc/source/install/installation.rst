.. _installation:

Installation
============

.. note::
    Before you can download or clone PyLag's source code, you must first register to use PyLag by following the instructions `here <http://www.pml.ac.uk/Modelling_at_PML/Access_Code>`_.

*PyLag* has been developed and tested within a UNIX/Linux environment, and the following instructions assume the user is working within a similar environment. The code is distributed in two distinct packages. The first contains the *PyLag* source code; the second a set of tools to help with setting up and analysing *PyLag* simulations.

.. _download:

Download
--------

The preferred means of accessing the code is through the source code management system `git <https://git-scm.com/>`_, which is free open-source software available for most common computing platforms. By using *git*, users can easily stay up to date with the latest *PyLag* release, report issues or track bugfixes. For *PyLag*, *git* access is enabled through the *git* management system `GitLab <https://gitlab.ecosystem-modelling.pml.ac.uk>`_.

After you have registered to use the code, a *GitLab* account will be created for you. If you checked the *PyLag* tick-box on the registration page you will have been automatically added to the
`PyLag Group <https://gitlab.ecosystem-modelling.pml.ac.uk/groups/PyLag>`_.

When accessing the code using *git*, it is recommended that users use secure shell (SSH) to communicate with the *GitLab* server. This allows users to establish a secure connection between their computer and *GitLab*, and to easily pull and push repositories. To enable access via SSH, users must first add a public SSH key to *GitLab*.

To begin, first check for existing SSH keys on your computer:

.. code-block:: bash

    $ cd $HOME/.ssh
    $ ls -la

If neither id_rsa.pub or id_dsa.pub are listed, a new SSH key must first be generated. This can be achieved using the following commands:

.. code-block:: bash

    $ ssh-keygen -t rsa -C "username@email.com"
    $ ssh-add $HOME/.ssh/id_rsa

Now add the SSH key to GitLab. To do this, first copy the key to the clipboard:

.. code-block:: bash

    $ xclip -sel clip < $HOME/.ssh/id_rsa.pub

Then in a web browser, login to `GitLab <https://gitlab.ecosystem-modelling.pml.ac.uk>`_, click on *Profile settings* (third icon from the left in the top right hand corner of the GitLab webpage), and then the *SSH Keys* tab. Click *Add SSH Key*, then enter a title (e.g. *Work PC*), and paste the key from the clipboard into *Key*. Finally, click *Add key*, which will save the new key.

With SSH access setup, you can now clone the *PyLag* repository:

.. code-block:: bash

    $ mkdir -p $HOME/code/git/PyLag && cd $HOME/code/git/PyLag
    $ git clone https://gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag.git>
    $ git clone https://gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag-tools.git>


If you don't want to use git to access the code, you can always grab a copy by downloading and unpacking tarballs of the two repositories.


.. _pipinstall:

Installation using pip
----------------------

The cleanest way to install *PyLag* and *PyLag-tools* is by using  `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ to create two new virtual environments, then using the python package manager `pip <https://pip.pypa.io/en/stable/>`_ to install the two packages within the new virtual environments. However, both packages can also be installed locally by passing the `--user` flag to *pip*. The use of *sudo* -- which would allow *PyLag* and *PyLag-tools* to be installed at the system level -- is strongly discouraged.

PyLag
`````

At the time of writing, *PyLag* itself will only work with the `Python 2.7 interpreter <https://www.python.org/download/releases/2.7>`_. As this is still the default interpreter on most common Linux distributions, it typically won't need installing separately. To begin installing *PyLag*, use `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ to configure a new virtual environment:

.. code-block:: bash

    $ mkvirtualenv -p /usr/bin/python --no-site-packages pylag

Python includes a dependency on the python package `MPI for Python <https://mpi4py.readthedocs.io/en/stable/>`_, which is used to facilitate running particle tracking simulations in parallel. To install *MPI for Python*, it is first necessary to ensure that you have a working MPI implementation on your system, and that all paths to MPI libraries and header files have been correctly set. You must use your Linux package manager to install a working MPI Implementation. On my laptop running Fedora 27, the following commands suffice:

.. code-block:: bash

    sudo dnf install -y mpich mpich-devel python2-mpich

Alternatively, if it is available via your package manager, you can install `python2-mpi4py` at the system level, which should automatically install all necessary MPI dependencies.

On my machine, *mpich* is enabled using the module command, which correctly sets environmental paths to the *mpich* MPI libraries and header files:

.. code-block:: bash

   module load mpi/mpich-x86_64

.. note::
    If this fails, try using `module avail` to list available MPI modules.

With MPI support enabled, it is now possible to install *PyLag* within the new virtual environment:

.. code-block:: bash

    $ cd $HOME/code/git/PyLag/PyLag
    $ pip install -r requirements.txt
    $ pip install .

To test that *PyLag* has been correctly installed, type:

.. code-block:: bash

    $ python -c "import pylag"

which should exit without error. To install *PyLag* locally is arguably easier, since if *MPI for Python* is installed at the system level, you will not have to worry about updating your paths for MPI (this is required in the virtual environment, since *MPI for Python* is built from source after being pulled down from *PyPI*).If you do install *PyLag* locally, simply pass the `--user` flag to both invocations of `pip install`.


PyLag-tools
```````````

The installation of *PyLag-tools* within a new virtual environment is complicated by the fact that *Pylag* leverages `basemap <https://matplotlib.org/basemap/>`_ to map certain outputs, and that *basemap* is not available via PyPi and must be installed manually. While *basemap* can be installed within a new virtual environment, it is typically not a one step process. If *PyLag-tools* is installed within a new virtual environment and *basemap* is not available, then all functionality that leverages *basemap* will be disabled. If you would like to use basemap, by far the easiest way to install it is by using your package manager. For example, in Fedora type:

.. code-block:: bash

    $ sudo dnf install python3-basemap

where we have specified that we want the *Python3* version.

.. note::
    Matploblib's *basemap* is being phased out in favour of `Cartopy <https://matplotlib.org/basemap/users/intro.html>`_. In the future, *PyLag-tools* will transition to using *Cartopy* too.

Then, to use *basemap's* functionality within a new virtual environment, you can tell virtualenv to add system packages to your *Python* path:

.. code-block:: bash

    $ mkvirtualenv -p /usr/bin/python3 --system-site-packages pylagtools

You can then install *PyLag-tools* using the following commands:

.. code-block:: bash

    $ cd $HOME/code/git/PyLag/PyLag-tools
    $ pip install -r requirements.txt

To test that *PyLag-tools* has been correctly installed, type:

.. code-block:: bash

    $ python -c "import pylagtools"

within the virtual environment. The command should exit without error. As with *PyLag*, to install *PyLag-tools* locally you just need to pass the `--user` flag to `pip install`.

.. note::
    *PyLag-tools* leverages the functionality of a package called `PyQt-fit <https://pyqt-fit.readthedocs.io/en/latest/index.html>`_, which has some nice features with respect to kernel density estimators that I have not found elsewhere. However, the package appears to no longer be maintained and the version available from PyPI has fallen out of sync with more recent releases of packages it depends on resulting in it failing to build out of the box. For these reasons, the version of *PyQt-fit* used by *PyLag-tools* is a fork of the original project code, with a couple of small fixes that allow it to be installed.

.. note::
    If you attempt to install *PyLag-tools* using a *Python2* interpreter, you will typically hit upon an error that arises in the package `PyFVCOM <https://gitlab.ecosystem-modelling.pml.ac.uk/fvcom/pyfvcom>`_. PyLag-tools leverages some of *PyFVCOM's* functionality for working with *FVCOM* datasets, and *PyFVCOM* has stopped supporting the *Python2* interpreter.
