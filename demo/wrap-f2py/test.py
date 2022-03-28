from mpi4pyve import MPI
import helloworld as hw

null = MPI.COMM_NULL
fnull = null.py2f()
hw.sayhello(fnull)

comm = MPI.COMM_WORLD
fcomm = comm.py2f()
hw.sayhello(fcomm)

try:
    hw.sayhello(list())
except:
    pass
else:
    assert 0, "exception not raised"
