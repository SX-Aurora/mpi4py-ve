from mpi4pyve import rc
del rc.initialize
del rc.threads
del rc.thread_level
del rc.finalize

from mpi4pyve import MPI
assert MPI.Is_initialized()
assert not MPI.Is_finalized()

import sys
name, _ = MPI.get_vendor()
if name == 'MPICH':
    assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
if name == 'MPICH2' and sys.platform[:3] != 'win':
    assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
