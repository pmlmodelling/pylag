.. _installation:

Installation
============

For end users, the easiest way to install *PyLag* is using *Conda*. This will install *PyLag* and all its dependencies into a clean environment and make it available to use. Developers who wish to work with the source code directly should clone *PyLag's* git repository from `GitHub <https://github.com/jimc101/PyLag>`_ and install it manually. Both use cases are described below.

.. note::
        *PyLag* has been developed and tested within a UNIX/Linux environment, and the following instructions assume the user is working in a similar environment.

.. _users:

Installation using Conda
------------------------

First `install miniconda3 <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ in a location of your choosing. Then, activate *Conda*, ensure conda is up to data, and add the channels *conda-forge*, *geo-down-under* and *JimClark*. The channel *geo-down-under* is needed for one of PyLag's dependencies. The channel *JimClark* is a temporary distribution channel for *PyLag*. For example:

.. code-block:: bash

    $ source /opt/miniconda/miniconda3/bin/activate
    $ conda update conda
    $ conda config --append channels conda-forge
    $ conda config --append channels geo-down-under
    $ conda config --append channels JimClark

The above code assumes *miniconda3* was installed into the directory ``/opt/miniconda``, once the appropriate write permissions have been set. The default behaviour is to install *miniconda3* into your home directory. This is, of course, also fine.

With *miniconda3* installed and configured, create a new environment in which to install *PyLag* using the following commands:

.. code-block:: bash

    $ conda create -n particles python=3.8
    $ conda activate particles

Finally, install *PyLag*:

.. code-block:: bash

    (particles) $ conda install -n particles -c JimClark pylag

To test that *PyLag* has been correctly installed, type:

.. code-block:: bash

    (particles) $ python -c "import pylag"

which should exit without error.

.. _developers:


Building from source
--------------------

Developers who wish to work with the source code directly should first clone *PyLag's* git repository from `GitHub <https://github.com/jimc101/PyLag>`_. With SSH access setup, you can clone the repository using the following commands:

.. code-block:: bash

    $ mkdir -p $HOME/code/git/PyLag && cd $HOME/code/git/PyLag
    $ git clone https://github.com/jimc101/PyLag.git

The cleanest and safest way of installing *PyLag's* dependencies is using *Conda*. Following steps similar to those described above, we can configure a new *Conda* environment so:

.. code-block:: bash

    $ source /opt/miniconda/miniconda3/bin/activate
    $ conda config --append channels conda-forge
    $ conda config --append channels geo-down-under
    $ conda install conda-build conda-verify

The new step here is the installation of conda-build and conda-verify. Note we don't add the JimClark channel in order to avoid conda installing pylag from Anaconda cloud. Next, create a new environment as above:

.. code-block:: bash

    $ conda create -n particles python=3.8
    $ conda activate particles

And finally, in the PyLag source code directory, build and install *PyLag*.

.. code-block:: bash

    (particles) $ cd $HOME/code/git/PyLag/PyLag
    (particles) $ conda build . --numpy 1.20
    (particles) $ conda install -n particles --use-local pylag

Occasionally, when building *PyLag* this way, users have hit upon clashes with locally installed packages. To get around this problem, you may find it useful to add the following aliases to your bashrc file, which you can use to activate and deactivate *Conda*:

.. code-block:: bash

    alias start_conda='export PYTHONNOUSERSITE=True && source /opt/miniconda/miniconda3/bin/activate'
    alias stop_conda='unset PYTHONNOUSERSITE && conda deactivate'

The *Conda* build process is quite long, and it doesn't lend itself to rapid build-install-test cycles. If you find yourself wanting to perform repeated builds, it is recommended you build using *Conda* on the first attempt, as described above. This will ensure PyLag's dependencies are correctly installed. Then, after this, you can install *PyLag* using *pip* like so:

.. code-block:: bash

    (particles) $ cd $HOME/code/git/PyLag/PyLag
    (particles) $ pip install .


.. _alternatives:

Alternative installation methods
--------------------------------

In principle, there are several other ways *PyLag* can be installed. For example, using `virtualenv <https://virtualenv.pypa.io/en/stable/>`_; or by using *pip* to perform a local install with the ``--user`` flag. The main thing to watch out for with these other methods is dependency issues. In particular, make sure you have *Cython* and *NumPy* installed already (e.g. using *pip* or *dnf*). Furthermore, *Conda* correctly configures your environment to make it possible to run *PyLag* in serial or parallel modes. When not using *Conda*, you will likely have to configure your environment to support parallel execution (and, in-fact, installation). This is because *PyLag* includes a dependency on the python package `MPI for Python <https://mpi4py.readthedocs.io/en/stable/>`_. To install *MPI for Python*, it is first necessary to ensure that you have a working MPI implementation on your system, and that all paths to MPI libraries and header files have been correctly set. You must use your Linux package manager to install a working MPI Implementation. On my laptop running Fedora 31, the following commands suffice:

.. code-block:: bash

   $ sudo dnf install -y openmpi openmpi-devel python3-openmpi

On my machine, *openmpi* is enabled using the module command, which correctly sets environment paths to the *openmpi* MPI libraries and header files:

.. code-block:: bash

   $ module load mpi/openmpi-x86_64

If running the above command fails with the system saying it is unable to find the *module* command, first use your package manager (e.g. *dnf*) to  ensure that the *environment-modules* package is installed. After installing it, you will need to open a new terminal. If it is still not found, try running:

.. code-block:: bash

    $ source /etc/profile.d/modules.sh

first. If this fixes the problem, you can add the above command to your *.bashrc* file.

.. note::
    The use of *sudo* -- which would allow *PyLag* to be installed at the system level -- is strongly discouraged.

Building the docs
-----------------

To build PyLag's documentation, a number of extra dependencies are required. These are not packaged with *PyLag* by default in order to keep the base installation slim and easier to manage. If you would like to build the documentation, the extra dependencies can be installed using *conda* or *pip*. The following commands use conda to install all the extra dependencies in the conda environment already created:

.. code-block:: bash

   (particles) $ conda install -n particles sphinx nbsphinx sphinx_rtd_theme sphinxcontrib-napoleon jupyter \
                 jupyter_client ipykernel ffmpeg seapy cmocean matplotlib shapely cartopy
   (particles) $ conda install -c JimClark -n particles PyFVCOM PyQT-fit

If you haven't added the JimClark channel you will need to do this before installing PyFVCOM and PyQT-fit.
