==================
mpi4py-ve tutorial
==================

Overview
--------

* *mpi4py-ve* is an extension to *mpi4py*, which provides Python bindings for the Message Passing Interface (MPI).

* The current version of *mpi4py-ve* communicates via x86 servers (Vector Host).

* For details of the *mpi4py-ve* API, refer to the `mpi4py manual <https://mpi4py.readthedocs.io/en/stable/index.html>`_.

This package supports:

* Convenient communication of any *picklable* Python object

  + point-to-point (send & receive)
  + collective (broadcast, scatter & gather, reductions)

* Communication of Python object exposing the *Python buffer
  interface* (NLCPy arrays, builtin bytes/string/array objects)

  + point-to-point (blocking/nonbloking/persistent send & receive)
  + collective (broadcast, block/vector scatter & gather, reductions)

* Process groups and communication domains

  + Creation of new intra/inter communicators
  + Cartesian & graph topologies

* Parallel input/output:

  + read & write
  + blocking/nonbloking & collective/noncollective
  + individual/shared file pointers & explicit offset

This package has NOT supported the following functions yet:

* One-sided operations

  + remote memory access (put, get, accumulate)
  + passive target syncronization (start/complete & post/wait)
  + active target syncronization (lock & unlock)

* Dynamic process management

  + spawn & spawn multiple
  + accept/connect
  + name publishing & lookup


List of Supprted Functions
--------------------------

A list of supported functions is shown below.

* MPI.Comm Class (Communicator)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Allgather             Gather to All, gather data from all processes and distribute it to all other processes in a group.
Allgatherv            Gather to All Vector, gather data from all processes and distribute it to all other processes in a group providing different amount of data and displacements.
Allreduce             Reduce to All.
Alltoall              All to All Scatter/Gather, send data from all to all processes in a group.
Alltoallv             All to All Scatter/Gather Vector, send data from all to all processes in a group providing different amount of data and displacements.
Alltoallw             Generalized All-to-All communication allowing different counts, displacements and datatypes for each partner.
Bcast                 Broadcast a message from one process to all other processes in a group.
Bsend                 Blocking send in buffered mode.
Bsend_init            Persistent request for a send in buffered mode.
Gather                Gather together values from a group of processes.
Gatherv               Gather Vector, gather data to one process from all other processes in a group providing different amount of data and displacements at the receiving sides.
Iallgather            Nonblocking Gather to All.
Iallgatherv           Nonblocking Gather to All Vector.
Iallreduce            Nonblocking Reduce to All.
Ialltoall             Nonblocking All to All Scatter/Gather.
Ialltoallv            Nonblocking All to All Scatter/Gather Vector.
Ialltoallw            Nonblocking Generalized All-to-All.
Ibcast                Nonblocking Broadcast.
Ibsend                Nonblocking send in buffered mode.
Igather               Nonblocking Gather.
Igatherv              Nonblocking Gather Vector.
Irecv                 Nonblocking receive.
Ireduce               Nonblocking Reduce to Root.
Ireduce_scatter       Nonblocking Reduce-Scatter (vector version).
Ireduce_scatter_block Nonblocking Reduce-Scatter Block (regular, non-vector version).
Irsend                Nonblocking send in ready mode.
Iscatter              Nonblocking Scatter.
Iscatterv             Nonblocking Scatter Vector.
Isend                 Nonblocking send.
Issend                Nonblocking send in synchronous mode.
Recv                  Blocking receive.
Recv_init             Create a persistent request for a receive.
Reduce                Reduce to Root.
Reduce_scatter        Reduce-Scatter (vector version).
Reduce_scatter_block  Reduce-Scatter Block (regular, non-vector version).
Rsend                 Blocking send in ready mode.
Rsend_init            Persistent request for a send in ready mode.
Scatter               Scatter data from one process to all other processes in a group.
Scatterv              Scatter Vector, scatter data from one process to all other processes in a group providing different amount of data and displacements at the sending side.
Send                  Blocking send.
Send_init             Create a persistent request for a standard send.
Sendrecv              Send and receive a message.
Sendrecv_replace      Send and receive a message.
Ssend                 Blocking send in synchronous mode.
Ssend_init            Persistent request for a send in synchronous mode.
allgather             Gather to All.
allreduce             Reduce to All.
alltoall              All to All Scatter/Gather.
bcast                 Broadcast.
bsend                 Send in buffered mode.
gather                Gather.
ibsend                Nonblocking send in buffered mode.
irecv                 Nonblocking receive.
isend                 Nonblocking send.
issend                Nonblocking send in synchronous mode.
recv                  Receive.
reduce                Reduce to Root.
scatter               Scatter.
send                  Send.
sendrecv              Send and Receive.
ssend                 Send in synchronous mode.
===================== ===============================================================================================================================================================

* MPI.Intracomm Class (Intracommunicator)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Exscan                Exclusive Scan.
Iexscan               Inclusive Scan.
Iscan                 Inclusive Scan.
Scan                  Inclusive Scan.
exscan                Exclusive Scan.
scan                  Inclusive Scan.
===================== ===============================================================================================================================================================

* MPI.Topocomm Class (Topology intracommunicator)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Ineighbor_allgather   Nonblocking Neighbor Gather to All.
Ineighbor_allgatherv  Nonblocking Neighbor Gather to All Vector.
Ineighbor_alltoall    Nonblocking Neighbor All-to-All.
Ineighbor_alltoallv   Nonblocking Neighbor All-to-All Vector.
Ineighbor_alltoallw   Nonblocking Neighbor All-to-All Generalized.
Neighbor_allgather    Neighbor Gather to All.
Neighbor_allgatherv   Neighbor Gather to All Vector.
Neighbor_alltoall     Neighbor All-to-All.
Neighbor_alltoallv    Neighbor All-to-All Vector.
Neighbor_alltoallw    Neighbor All-to-All Generalized.
neighbor_allgather    Neighbor Gather to All.
neighbor_alltoall     Neighbor All to All Scatter/Gather.
===================== ===============================================================================================================================================================

* MPI (Miscellanea)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Attach_buffer         Attach a user-provided buffer for sending in buffered mode.
===================== ===============================================================================================================================================================

* MPI.Request Class (Request handle)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Wait                  Wait for a send or receive to complete
Waitall               Wait for all previously initiated requests to complete.
Waitany               Wait for any previously initiated request to complete.
Waitsome              Wait for some previously initiated requests to complete.
wait                  Wait for a send or receive to complete.
waitall               Wait for all previously initiated requests to complete.
waitany               Wait for any previously initiated request to complete.
===================== ===============================================================================================================================================================

* MPI.Message Class (Communication / Matched message handle)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Irecv                 Nonblocking receive of matched message.
Recv                  Blocking receive of matched message.
irecv                 Nonblocking receive of matched message.
recv                  Blocking receive of matched message.
===================== ===============================================================================================================================================================

* MPI.Op Class (Ancillay / Operation object)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Reduce_local          Apply a reduction operator to local data.  
===================== ===============================================================================================================================================================

* MPI.Datatype Class (Ancillay / Datatype object)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Pack                  Pack into contiguous memory according to datatype.
Pack_external         Pack into contiguous memory according to datatype, using a portable data representation (external32).
Unpack                Unpack from contiguous memory according to datatype.
Unpack_external       Unpack from contiguous memory according to datatype, using a portable data representation (external32).
===================== ===============================================================================================================================================================

* MPI.File Class (Parallel input/output)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Iread                 Nonblocking read using individual file pointer.
Iread_all             Nonblocking collective read using individual file pointer.
Iread_at              Nonblocking read using explicit offset.
Iread_at_all          Nonblocking collective read using explicit offset.
Iread_shared          Nonblocking read using shared file pointer.
Iwrite                Nonblocking write using individual file pointer.
Iwrite_all            Nonblocking collective write using individual file pointer.
Iwrite_at             Nonblocking write using explicit offset.
Iwrite_at_all         Nonblocking collective write using explicit offset.
Iwrite_shared         Nonblocking write using shared file pointer.
Read                  Read using individual file pointer.
Read_all              Collective read using individual file pointer.
Read_all_begin        Start a split collective read using individual file pointer.
Read_all_end          Complete a split collective read using individual file pointer.
Read_at               Read using explicit offset.
Read_at_all           Collective read using explicit offset.
Read_at_all_begin     Start a split collective read using explict offset.
Read_at_all_end       Complete a split collective read using explict offset.
Read_ordered          Collective read using shared file pointer.
Read_ordered_begin    Start a split collective read using shared file pointer.
Read_ordered_end      Complete a split collective read using shared file pointer.
Read_shared           Read using shared file pointer.
Write                 Write using individual file pointer.
Write_all             Collective write using individual file pointer.
Write_all_begin       Start a split collective write using individual file pointer.
Write_all_end         Complete a split collective write using individual file pointer.
Write_at              Write using explicit offset.
Write_at_all          Collective write using explicit offset.
Write_at_all_begin    Start a split collective write using explict offset.
Write_at_all_end      Complete a split collective write using explict offset.
Write_ordered         Collective write using shared file pointer.
Write_ordered_begin   Start a split collective write using shared file pointer.
Write_ordered_end     Complete a split collective write using shared file pointer.
Write_shared          Write using shared file pointer.
===================== ===============================================================================================================================================================

List of Unsupprted Functions
----------------------------

The current version of *mpi4py-ve* does not support the following functions. Please note that "NotImplementedError" occurs if your Python script calls them.

* MPI.Comm Class (Communicator)

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Accept                Accept a request to form a new intercommunicator.
Connect               Make a request to form a new intercommunicator.
Close_port            Close a port.
Join                  Create a intercommunicator by joining two processes connected by a socket.
Lookup_name           Lookup a port name given a service name.
Open_port             Return an address that can be used to establish connections between groups of MPI processes.
Publish_name          Publish a service name.
Unpublish_name        Unpublish a service name.
===================== ===============================================================================================================================================================

* MPI.Win Class (One-sided operations) 

===================== ===============================================================================================================================================================
Name                  Summary
===================== ===============================================================================================================================================================
Accumulate            Accumulate data into the target process.
Compare_and_swap      Perform one-sided atomic compare-and-swap.
Fetch_and_op          Perform one-sided read-modify-write.
Get                   Get data from a memory window on a remote process.
Get_accumulate        Fetch-and-accumulate data into the target process.
Put                   Put data into a memory window on a remote process.
Raccumulate           Fetch-and-accumulate data into the target process.
Rget                  Get data from a memory window on a remote process.
Rget_accumulate       Accumulate data into the target process using remote memory access.
Rput                  Put data into a memory window on a remote process.
===================== ===============================================================================================================================================================

Exception Handling
------------------

This section describes how to handle unhandled exceptions.
Assume this code is stored in a standard Python script file and run with mpirun in two or more processes.

**ZeroDivisionError.py**

.. code-block:: python

    from mpi4pyve import MPI
    assert MPI.COMM_WORLD.Get_size() > 1
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        1/0
        MPI.COMM_WORLD.send(None, dest=1, tag=42)
    elif rank == 1:
        MPI.COMM_WORLD.recv(source=0, tag=42)

::

    $ mpirun -veo -np 2 $(which python) ZeroDivisionError.py

Process 0 raises **ZeroDivisionError** exception before performing a send call to process 1. As the exception is not handled, the Python interpreter running in process 0 will proceed to exit with non-zero status. However, as *mpi4py-ve* installed a finalizer hook to call *MPI_Finalize()* before exit, process 0 will block waiting for other processes to also enter the *MPI_Finalize()* call. Meanwhile, process 1 will block waiting for a message to arrive from process 0, thus never reaching to *MPI_Finalize()*. The whole MPI execution environment is irremediably in a deadlock state.

To alleviate this issue, *mpi4py-ve* offers a simple, alternative command line execution mechanism based on using the `-m <https://docs.python.org/3/using/cmdline.html#using-on-cmdline>`_ flag and implemented with the *runpy* module. To use this features, Python code should be run passing **-m mpi4pyve** in the command line invoking the Python interpreter. In case of unhandled exceptions, the finalizer hook will call *MPI_Abort()* on the *MPI_COMM_WORLD* communicator, thus effectively aborting the MPI execution environment.

    ::

    $ mpirun -veo -np 2 $(which python) -m mpi4pyve ZeroDivisionError.py


This is a mimic of the option **-m mpi4py** described in the `mpi4py manual (mpi4py.run) <https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html>`_.
