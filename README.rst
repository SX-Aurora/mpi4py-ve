#########
mpi4py-ve 
#########

*mpi4py-ve* is an extension to *mpi4py*, which provides Python bindings for the Message Passing Interface (MPI).
This package also supports to communicate array objects of `NLCPy <https://sxauroratsubasa.sakura.ne.jp/documents/nlcpy/en/>`_ (nlcpy.ndarray) between MPI processes on x86 servers of SX-Aurora TSUBASA systems.
Combining NLCPy with *mpi4py-ve* enables Python scripts to utilize multi-VE computing power.
The current version of *mpi4py-ve* is based on *mpi4py* version 3.0.3.
For details of API references, please refer to `mpi4py manual <https://mpi4py.readthedocs.io/en/stable/>`_.

************
Requirements
************

Before the installation, the following components are required to be installed on your x86 Node of SX-Aurora TSUBASA.

- `Alternative VE Offloading (AVEO) <https://sxauroratsubasa.sakura.ne.jp/documents/veos/en/aveo/index.html>`_
	- required version: >= 2.13.0

- `NEC MPI <https://sxauroratsubasa.sakura.ne.jp/documents/mpi/g2am01e-NEC_MPI_User_Guide_en/frame.html>`_
	- required NEC MPI version: > 2.22.0 (for Mellanox OFED 4.x) or >= 3.1.0 (for Mellanox OFED 5.x)

- `Python <https://www.python.org/>`_
        - required version: 3.6, 3.7, or 3.8

- `NumPy <https://www.numpy.org/>`_
        - required version: v1.17, v1.18, v1.19, or v1.20

- `NLC(optional) <https://sxauroratsubasa.sakura.ne.jp/documents/sdk/SDK_NLC/UsersGuide/main/en/index.html>`_
	- required version: >= 2.3.0

- `NLCPy(optional) <https://sxauroratsubasa.sakura.ne.jp/documents/nlcpy/en/>`_
        - required version: >= 2.2.0

Since December 2022, mpi4py-ve has been provided as a software of NEC SDK (NEC Software Development Kit for Vector Engine).
If NEC SDK on your machine has been properly installed or updated after that, mpi4py-ve is available by using /usr/bin/python3 command.

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

The shared objects for Vector Host, which are included in the wheel package, are compiled by gcc 4.8.5 and tested by using following softwares:
    +---------+--------------------+
    | NEC MPI | v2.24.0 and V3.3.0 |
    +---------+--------------------+
    | NumPy   | v1.19.5            |
    +---------+--------------------+
    | NLCPy   | v3.0.0             |
    +---------+--------------------+

***********************************
Install from source (with building)
***********************************

Before building this package, you need to execute the environment setup script *necmpivars.sh* or *necmpivars.csh* once advance.

* When using *sh* or its variant:

    **For VE30**
    
        ::

        $ source /opt/nec/ve3/mpi/X.X.X/bin/necmpivars.sh

    **For VE20, VE10, or VE10E**
    
        ::
        
        $ source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.sh

* When using *csh* or its variant:

    **For VE30**
    
        ::

        % source /opt/nec/ve3/mpi/X.X.X/bin/necmpivars.csh

    **For VE20, VE10, or VE10E**
    
        ::
        
        % source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.csh

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

    **For VE30**

        ::

        $ source /opt/nec/ve3/mpi/X.X.X/bin/necmpivars.sh gnu 4.8.5

    **For VE20, VE10, or VE10E**

        ::
        
        $ source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.sh gnu 4.8.5

* When using *csh* or its variant:

    **For VE30**

        ::

        % source /opt/nec/ve3/mpi/X.X.X/bin/necmpivars.csh gnu 4.8.5
    
    **For VE20, VE10, or VE10E**
    
        ::
        
        % source /opt/nec/ve/mpi/X.X.X/bin/necmpivars.csh gnu 4.8.5

Here, X.X.X denotes the version number of NEC MPI.

When using the *mpirun* command:

    ::

    $ mpirun -veo -np N $(which python) sample.py

| Here, N is the number of MPI processes that are created on an x86 server.
| NEC MPI 2.21.0 or later supports the environment variable `NMPI_USE_COMMAND_SEARCH_PATH`.
| If `NMPI_USE_COMMAND_SEARCH_PATH` is set to `ON` and the Python command path is added to the environment variable PATH, you do not have to specify with the full path.

    ::

    $ export NMPI_USE_COMMAND_SEARCH_PATH=ON
    $ mpirun -veo -np N python sample.py

| For details of mpirun command, refer to `NEC MPI User's Guide <https://sxauroratsubasa.sakura.ne.jp/documents/mpi/g2am01e-NEC_MPI_User_Guide_en/frame.html>`_.

******************
Execution Examples
******************

The following examples show how to launch MPI programs that use mpi4py-ve and NLCPy on the SX-Aurora TSUBASA.

| *ncore* : Number of cores per VE.
| a.py: Python script using mpi4py-ve and NLCPy.
| 

* Interactive Execution

  * Execution on one VE

    Example of using 4 processes on local VH and 4 VE processes (*ncore* / 4 OpenMP parallel per process) on VE#0 of local VH

    ::

      $ mpirun -veo -np 4 python a.py

  * Execution on multiple VEs on a VH

    Example of using 4 processes on local VH and 4 VE processes (1 process per VE, *ncore* OpenMP parallel per process) on VE#0 to VE#3 of local VH

    ::

      $ VE_NLCPY_NODELIST=0,1,2,3 mpirun -veo -np 4 python a.py


    Example of using 32 processes on local VH and 32 VE processes (8 processes per VE, *ncore* / 8 OpenMP parallel per process) on VE#0 to VE# 3 of local VH

    ::

      $ VE_NLCPY_NODELIST=0,1,2,3 mpirun -veo -np 32 python a.py

  * Execution on multiple VEs on multiple VHs

    Example of using a total of 32 processes on two VHs host1 and host2, and a total of 32 VE processes on VE#0 and VE#1 of each VH (8 processes per VE, *ncore* / 8 OpenMP parallel per process)

    ::

      $ VE_NLCPY_NODELIST=0,1 mpirun -hosts host1,host2 -veo -np 32 python a.py

* NQSV Request Execution

  * Execution on a specific VH, on a VE

    Example of using 32 processes on logical VH#0 and 32 VE processes on logical VE#0 to logical VE#3 on logical VH#0 (8 processes per VE, *ncore* / 8 OpenMP parallel per process)

    ::

      #PBS -T necmpi
      #PBS -b 2 # The number of logical hosts
      #PBS --venum-lhost=4 # The number of VEs per logical host
      #PBS --cpunum-lhost=32 # The number of CPUs per logical host
      
      source /opt/nec/ve/mpi/2.22.0/bin/necmpivars.sh
      export NMPI_USE_COMMAND_SEARCH_PATH=ON
      mpirun -host 0 -veo -np 32 python a.py

  * Execution on a specific VH, on a specific VE

    Example of using 16 processes on logical VH#0, 16 VE processes in total on logical VE#0 and logical VE#3 on logical VH#0 (8 processes per VE, *ncore* / 8 OpenMP parallel per process)

    ::

      #PBS -T necmpi
      #PBS -b 2 # The number of logical hosts
      #PBS --venum-lhost=4 # The number of VEs per logical host
      #PBS --cpunum-lhost=16 # The number of CPUs per logical host
      
      source /opt/nec/ve/mpi/2.22.0/bin/necmpivars.sh
      export NMPI_USE_COMMAND_SEARCH_PATH=ON
      VE_NLCPY_NODELIST=0,3 mpirun -host 0 -veo -np 16 python a.py

  * Execution on all assigned VEs

    Example of using 32 processes in total on 4 VHs and using 32 VE processes in total from logical VE#0 to logical VE#7 on each of VHs (1 process per VE, *ncore* OpenMP parallel per process).

    ::

      #PBS -T necmpi
      #PBS -b 4 # The number of logical hosts
      #PBS --venum-lhost=8 # The number of VEs per logical host
      #PBS --cpunum-lhost=8 # The number of CPUs per logical host
      #PBS --use-hca=2 # The number of HCAs
      
      source /opt/nec/ve/mpi/2.22.0/bin/necmpivars.sh
      export NMPI_USE_COMMAND_SEARCH_PATH=ON
      mpirun -veo -np 32 python a.py

*********
Profiling
*********
NEC MPI provides the facility of displaying MPI communication information. 
There are two formats of MPI communication information available as follows:

+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+ 
| Reduced Format  | The maximum, minimum, and average values of MPI communication information of all MPI processes are displayed.                                                                        |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Extended Format | MPI communication information of each MPI process is displayed in the ascending order of their ranks in the communicator MPI_COMM_WORLD after the information in the reduced format. |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

You can control the display and format of MPI communication information by setting the environment variable NMPI_COMMINF at runtime as shown in the following table.

The Settings of NMPI_COMMINF:

+--------------+-----------------------+ 
| NMPI_COMMINF | Displayed Information |
+--------------+-----------------------+
| NO           | (Default) No Output   |
+--------------+-----------------------+
| YES          | Reduced Format        |
+--------------+-----------------------+
| ALL          | Extended Format       |
+--------------+-----------------------+

When using the *mpirun* command:

    ::

    $ export NMPI_COMMINF=ALL
    $ mpirun -veo -np N python sample.py

***************************************************
Use mpi4py-ve with homebrew classes (without NLCPy)
***************************************************

Below links would be useful to use *mpi4py-ve* with homebrew classes (without NLCPy):

* `use mpi4py-ve with homebrew classes (without NLCPy) <https://github.com/SX-Aurora/mpi4py-ve/blob/v1.0.0/docs/vai_spec_example.rst>`_

***************
Other Documents
***************

Below links would be useful to understand *mpi4py-ve* in more detail:

* `mpi4py-ve tutorial <https://github.com/SX-Aurora/mpi4py-ve/blob/v1.0.0/docs/index.rst>`_

***********
Restriction
***********
* The current version of *mpi4py-ve* does not support some functions that are listed in the section "List of Unsupported Functions" of `mpi4py-ve tutorial <https://github.com/SX-Aurora/mpi4py-ve/blob/v1.0.0/docs/index.rst>`_.
* Communication of type bool between NumPy and NLCPy will fail because of the different number of bytes.

*******
Notices
*******
* If you import NLCPy before calling MPI_Init()/MPI_Init_thread(), a runtime error will be raised.

    Not recommended usage: ::

        $ mpirun -veo -np 1 $(which python) -c "import nlcpy; from mpi4pyve import MPI"
        RuntimeError: NLCPy must be import after MPI initialization

    Recommended usage: ::

        $ mpirun -veo -np 1 $(which python) -c "from mpi4pyve import MPI; import nlcpy" 

    MPI_Init() or MPI_Init_thread() is called when you import the MPI module from the mpi4pyve package.

* If you use the Lock/Lock_all function for one-sided communication using NLCPy array data, you need to put in NLCPy synchronization control.

    Synchronization usage:

    .. code-block:: python

        import mpi4pyve
        from mpi4pyve import MPI
        import nlcpy as vp

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        array = vp.array(0, dtype=int)

        if rank == 0:
            win_n = MPI.Win.Create(array,  comm=MPI.COMM_WORLD)
        else:
            win_n = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
        if rank == 0:
            array.fill(1)
            array.venode.synchronize()
            comm.Barrier()
        if rank != 0:
           comm.Barrier()
            win_n.Lock(MPI.LOCK_EXCLUSIVE, 0)
            win_n.Get([array, MPI.INT], 0)
            win_n.Unlock(0)
            assert array == 1
        comm.Barrier()
        win_n.Free()

*******
License
*******

| The 2-clause BSD license (see *LICENSE* file).
| *mpi4py-ve* is derived from mpi4py (see *LICENSE_DETAIL/LICENSE_DETAIL* file).
