from mpi4pyve import rc
rc.initialize = False

from mpi4pyve import MPI
assert not MPI.Is_initialized()
assert not MPI.Is_finalized()

MPI.Init()
assert MPI.Is_initialized()
assert not MPI.Is_finalized()

MPI.Finalize()
assert MPI.Is_initialized()
assert MPI.Is_finalized()
