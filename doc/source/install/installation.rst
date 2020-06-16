.. _installation:

Installation
============

For end users, the easiest way to install *PyLag* is using *Conda*. This will install *PyLag* and all its dependencies into a clean environment and make it available to use. Developers who wish to work with the source code directly should first register to access *PyLag's* git repository `here <http://www.pml.ac.uk/Modelling_at_PML/Access_Code>`_. Once registered, you will be able to clone a copy of *PyLag's* source code and install it manually. Both use cases are described below.

.. note::
        *PyLag* has been developed and tested within a UNIX/Linux environment, and the following instructions assume the user is working
        within a similar environment.

.. _users:

Installation using Conda
------------------------

First `install miniconda3 <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ in a location of your choosing. 
Then, activate *Conda* and add the channels *conda-forge* and *JimClark*. The latter is a temporary distribution channel for *PyLag*. For example:

.. code-block:: bash

    $ source /opt/miniconda/miniconda3/bin/activate
    $ conda config --append channels conda-forge
    $ conda config --append channels JimClark

The above code assumes *miniconda3* was installed into the directory ``/opt/miniconda``, once the appropriate write permissions have been set. The default behaviour is to install *miniconda3* into your home directory. This is, of course, also fine.

With *miniconda3* installed and configured, create a new environment in which to install *PyLag* using the following commands:

.. code-block:: bash

    $ conda create -n particles python=3.7
    $ conda activate particles

Finally, install *PyLag*:

.. code-block:: bash

    $ conda install -n particles -c JimClark pylag

To test that *PyLag* has been correctly installed, type:

.. code-block:: bash

    $ python -c "import pylag"

which should exit without error.

.. _developers:


Building from source
--------------------

Developers who wish to work with the source code directly should first register to access *PyLag's* git repository `here <http://www.pml.ac.uk/Modelling_at_PML/Access_Code>`_. The preferred means of accessing the source code is through the source code management system `git <https://git-scm.com/>`_, which is free open-source software available for most common computing platforms. By using *git*, users can easily stay up to date with the latest *PyLag* release, report issues or track bugfixes. For *PyLag*, *git* access is enabled through the *git* management system `GitLab <https://gitlab.ecosystem-modelling.pml.ac.uk>`_.

After you have registered to use the code, a *GitLab* account will be created for you. If you checked the *PyLag* tick-box on the registration page you will have been automatically added to the
`PyLag Group <https://gitlab.ecosystem-modelling.pml.ac.uk/groups/PyLag>`_.

When accessing the code using *git*, it is recommended that users use secure shell (SSH) to communicate with the *GitLab* server. This allows users to establish a secure connection between their computer and *GitLab*, and to easily pull and push repositories.

With SSH access setup, you can clone both repositories using the following commands:

.. code-block:: bash

    $ mkdir -p $HOME/code/git/PyLag && cd $HOME/code/git/PyLag
    $ git clone https://gitlab.ecosystem-modelling.pml.ac.uk/PyLag/PyLag.git>


If you don't want to use git to access the code, you can always grab a copy by downloading and unpacking tarballs of the two repositories. The cleanest and safest way of installing *PyLag's* dependencies is using *Conda*. Following steps similar to those described above, we can configure a new *Conda* environment so:

.. code-block:: bash

    $ source /opt/miniconda/miniconda3/bin/activate
    $ conda config --append channels conda-forge
    $ conda config --append channels JimClark
    $ conda install conda-build conda-verify

The only new step here is the installation of conda-build and conda-verify. Next, create a new environment as above:

.. code-block:: bash

    $ conda create -n particles python=3.7
    $ conda activate particles

And finally, in the PyLag source code directory, build and install *PyLag*.

.. code-block:: bash

    $ cd $HOME/code/git/PyLag/PyLag
    $ conda build .
    $ conda install -n particles --use-local pylag

Occsionally, when building *PyLag* this way, users have hit upon clashes with locally installed packages. To get around this problem, you may find it useful to add the following aliases to your bashrc file, which you can use to activate and deactivate *Conda*:

.. code-block:: bash

    alias start_conda='export PYTHONNOUSERSITE=True && source /opt/miniconda/miniconda3/bin/activate'
    alias stop_conda='unset PYTHONNOUSERSITE && conda deactivate'

The *Conda* build process is quite long, and it doesn't lend itself to rapid build-install-test cycles. If you find yourself wanting to perform repeated builds, it is recommended you build using *Conda* on the first attempt, as described above. This will ensure PyLag's dependencies are correctly installed. Then, after this, you can install *PyLag* using *pip* like so:

.. code-block:: bash

    $ cd $HOME/code/git/PyLag/PyLag
    $ pip install .


.. _alternatives:

Alternative installation methods
--------------------------------

In principle, there are several other ways *PyLag* can be installed. For example, using `virtualenv <https://virtualenv.pypa.io/en/stable/>`_; or by using *pip* to perform a local install with the ``--user`` flag. The main thing to watch out for with these other methods is dependency issues. In particular, *PyLag* leverages functionality within the `PyFVCOM <https://pypi.org/project/PyFVCOM/>`_ and `PyQt-fit <https://pyqt-fit.readthedocs.io/en/latest/index.html>`_ packages. When building using *Conda*, pre-built versions of theses packages are brought down and installed automatically. However, with custom installs, they may need to be installed separately. Furthermore, *Conda* correctly configures your environment to make it possible to run *PyLag* in serial or parallel modes. When not using *Conda*, you will likely have to configure your environment to support parallel execution (and, in-fact, installation).

This is because *PyLag* includes a dependency on the python package `MPI for Python <https://mpi4py.readthedocs.io/en/stable/>`_. To install *MPI for Python*, it is first necessary to ensure that you have a working MPI implementation on your system, and that all paths to MPI libraries and header files have been correctly set. You must use your Linux package manager to install a working MPI Implementation. On my laptop running Fedora 27, the following commands suffice:

.. code-block:: bash

   sudo dnf install -y openmpi python3-openmpi

Alternatively, if it is available via your package manager, you can install `python3-mpi4py` at the system level, which should automatically install all necessary MPI dependencies.

On my machine, *openmpi* is enabled using the module command, which correctly sets environmental paths to the *openmpi* MPI libraries and header files:

.. code-block:: bash

   module load mpi/openmpi-x86_64

.. note::
    The use of *sudo* -- which would allow *PyLag* to be installed at the system level -- is strongly discouraged.
