#########
mpi4py-ve 
#########

*mpi4py-ve* is an extension to *mpi4py*, which provides Python bindings for the Message Passing Interface (MPI).
This package also supports to communicate array objects of `NLCPy <https://www.hpc.nec/documents/nlcpy/en/>`_ (nlcpy.ndarray) between MPI processes on x86 servers of SX-Aurora TSUBASA systems.
Combining NLCPy with *mpi4py-ve* enables Python scripts to utilize multi-VE computing power.
The current version of *mpi4py-ve* is based on *mpi4py* version 3.0.3.
For details of API references, please refer to `mpi4py manual <https://mpi4py.readthedocs.io/en/stable/>`_.

************
Requirements
************

Before the installation, the following components are required to be installed on your x86 Node of SX-Aurora TSUBASA.

- `NEC SDK <https://www.hpc.nec/documents/guide/pdfs/InstallationGuide_E.pdf>`_
	- required NEC C/C++ compiler version: >= 3.2.1
	- required NLC version: >= 2.3.0

- `VEOS <https://www.hpc.nec/documents/veos/en/aveo/index.html>`_
	- required version: >= 2.11.1

- `NEC MPI <https://www.hpc.nec/documents/mpi/g2am01e-NEC_MPI_User_Guide_en/frame.html>`_
	- required NEC MPI version: >=  2.20.0

- `Python <https://www.python.org/>`_
        - required version: 3.6, 3.7, or 3.8

- `NLCPy <https://www.hpc.nec/documents/nlcpy/en/>`_
        - required version: v2.1.1

- `NumPy <https://www.numpy.org/>`_
        - required version: v1.17, v1.18, v1.19, or v1.20

******************
Install from wheel
******************

You can install *mpi4py-ve* by executing either of the following commands.

- Install from PyPI

    ::
 
    $ pip install mpi4py-ve

- Install from your local computer

    1. Download `the wheel package <https://github.com/SX-Aurora/mpi4py-ve/releases>`_ from GitHub.

    2. Put the wheel package to your any directory.

    3. Install the local wheel package via pip command.

        ::
 
        $ pip install <path_to_wheel>

The shared objects for Vector Engine, which are included in the wheel package, are compiled and tested by using following software:
    +-------------------+---------------+ 
    | NEC C/C++ Compiler| Version 3.2.1 |
    +-------------------+---------------+
    | NEC MPI           | v2.20.0       |
    +-------------------+---------------+
    | NumPy             | v1.19.2       |
    +-------------------+---------------+
    | NLCPy             | v2.1.1        |
    +-------------------+---------------+

***********************************
Install from source (with building)
***********************************

Before building this package, you need to execute the environment setup script *necmpivars.sh* or *necmpivars.csh* once advance.

* When using *sh* or its variant:

    ::

    $ source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.sh

* When using *csh* or its variant:

    ::

    $ source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.csh

Here, X.X.X denotes the version number of NEC MPI.

After that, execute the following commands:

    ::

    $ git clone https://github.com/SX-Aurora/mpi4py-ve.git
    $ cd mpi4py-ve
    $ python setup.py build --mpi=necmpi
    $ python setup.py install 

*******
Example
*******

**Transfer Array**

Transfers an NLCPy's ndarray from MPI rank 0 to 1 by using comm.Send() and comm.Recv():

.. code-block:: python

    from mpi4pyve import MPI
    import nlcpy as vp

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        x = vp.array([1,2,3], dtype=int)
        comm.Send(x, dest=1)

    elif rank == 1:
        y = vp.empty(3, dtype=int)
        comm.Recv(y, source=0)


**Sum of Numbers**

Sums the numbers locally, and reduces all the local sums to the root rank (rank=0):

.. code-block:: python

    from mpi4pyve import MPI
    import nlcpy as vp

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    N = 1000000000
    begin = N * rank // size
    end = N * (rank + 1) // size

    sendbuf = vp.arange(begin, end).sum()
    recvbuf = comm.reduce(sendbuf, MPI.SUM, root=0)

The following table shows the performance results[msec] on VE Type 20B:

+------+------+------+------+------+------+------+------+ 
| np=1 | np=2 | np=3 | np=4 | np=5 | np=6 | np=7 | np=8 |
+------+------+------+------+------+------+------+------+
| 35.8 | 19.0 | 12.6 | 10.1 |  8.1 |  7.0 |  6.0 |  5.5 |
+------+------+------+------+------+------+------+------+

*********
Execution
*********

When executing Python script using *mpi4py-ve*, use *mpirun* command of NEC MPI on an x86 server of SX-Aurora TSUBASA.
Before running the Python script, you need to execute the environment the following setup scripts once advance.

* When using *sh* or its variant:

    ::

    $ source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.sh gnu 4.8.5
    $ source /opt/nec/ve/nlc/Y.Y.Y/bin/nlcvars.sh

* When using *csh* or its variant:

    ::

    $ source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.csh gnu 4.8.5
    $ source /opt/nec/ve/nlc/Y.Y.Y/bin/nlcvars.csh

Here, X.X.X and Y.Y.Y denote the version number of NEC MPI and NLC, respectively.

When using the *mpirun* command:

    ::

    $ mpirun -vh -np N $(which python) sample.py

| Here, N is the number of MPI processes that are created on an x86 server.
| NEC MPI 2.21.0 or later supports the environment variable `NMPI_USE_COMMAND_SEARCH_PATH`.
| If `NMPI_USE_COMMAND_SEARCH_PATH` is set to `ON` and the Python command path is added to the environment variable PATH, you do not have to specify with the full path.

    ::

    $ export NMPI_USE_COMMAND_SEARCH_PATH=ON
    $ mpirun -vh -np N python sample.py

| For details of mpirun command, refer to `NEC MPI User's Guide <https://www.hpc.nec/documents/mpi/g2am01e-NEC_MPI_User_Guide_en/frame.html>`_.

***************
Other Documents
***************

Below links would be useful to understand *mpi4py-ve* in more detail:

* `mpi4py-ve tutorial <https://github.com/SX-Aurora/mpi4py-ve/blob/v0.1.0b1/docs/index.rst>`_

***********
Restriction
***********
* The value specified by np must not exceed the number of VE cards.
* The current version of *mpi4py-ve* does not support some functions that are listed in the section "List of Unsupported Functions" of `mpi4py-ve tutorial <https://github.com/SX-Aurora/mpi4py-ve/blob/v0.1.0b1/docs/index.rst>`_.

*******
License
*******

| The 2-clause BSD license (see *LICENSE* file).
| *mpi4py-ve* is derived from mpi4py (see *LICENSE_DETAIL/LICENSE_DETAIL* file).
