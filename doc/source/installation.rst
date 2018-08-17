.. _installation:

Installation
============

.. note::
    Before you can download or clone PyLag's source code, you must first register to use PyLag by following the instructions `here <http://www.pml.ac.uk/Modelling_at_PML/Access_Code>`_.

.. _requirements:

Requirements
------------

*PyLag* has been developed and tested within a UNIX/Linux environment, and the following instructions assume the user is working within a similar environment. The code is distributed in two distinct packages. The first contains the *PyLag* source code; the second a set of tools to help with setting up and analysing *PyLag* simulations. The two packages have separate dependencies that are described below.

PyLag
`````

At the time of writing, *PyLag* itself will only work with the `Python 2.7 interpreter <https://www.python.org/download/releases/2.7>`_. The *PyLag* package has a small number of direct dependencies, including:

* Cython
* NumPy
* netCDF4
* mpi4py
* ConfigParser
* natsort
* progressbar

Both packages include dependencies that are sometime best installed prior to the installation of the package itself. *PyLag* has a dependency on `Cython <http://www.cython.org>`_, as several modules have been implemented *Cython*. If *Cython* is not installed already, you can install it using your package manager. For example, in *Fedora* you can install *Cython* using the following command:

.. code-block:: bash

    $ sudo dnf install -y python2-cython

.. note::
    Different operating systems use different package managers. Recent releases of `Fedora <https://getfedora.org/>`_ use the `dnf package manager <https://fedoraproject.org/wiki/Dnf>`_ which is the successor to `yum <https://fedoraproject.org/wiki/Yum>`_. `ubuntu <http://www.ubuntu.com/>`_ uses the `apt <https://wiki.debian.org/Apt>`_ package manager .



Alternatively, Cython

PyLag-tools
```````````

*PyLag-tools* supports the `Python 3.6 interpreter <https://www.python.org/download/releases/3.6>`_, which is recommended for all analysis work that leaverages *PyLag-tools'* functionality.

The easiest way to install *PyLag* and *PyLag-tools* is using the python package manager `pip <https://pip.pypa.io/en/stable/>`_, which should automatically handle all python dependencies. If *pip* is not installed already, you can install it using your package manager.

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

The cleanest way to install *PyLag* and *PyLag-tools* is by using  `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ to create two new virtual environments. However, PyLag can also be installed locally by passing the *--user* flag to *pip*. The use of *sudo* -- which would allow *PyLag* and *PyLag-tools* to be installed at the system level -- is strongly discouraged.

To perform a local installation of *PyLag* given the above directory structure type:

.. code-block:: bash

    $ cd $HOME/code/git/PyLag/PyLag
    $ pip install --user -e .

*pip* will automatically search through *PyLag's* *Python* dependencies and try to install these if they are not found. To install PyLag-tools locally type:

.. code-block:: bash

    $ cd $HOME/code/git/PyLag/PyLag-tools
    $ pip3 install --user -e .


.. note::
    If you experience trouble invoking pip3 directly, try typing `$ python3 -m pip install --user -e .` instead.


You can check that PyLag and PyLag-tools have been successfully installed by running the
commands:

.. code-block:: bash

    $ python -c "import pylag"
    $ python3 -c "import pylagtools"


which should exit without error.
