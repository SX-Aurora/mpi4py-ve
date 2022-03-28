import os
from mpi4pyve import MPI

IS_MULTI_HOST = None

try:
    if IS_MULTI_HOST is None:
        comm = MPI.COMM_WORLD
        nodes = comm.allgather(os.environ['MPINODEID'])
        IS_MULTI_HOST = (len(list(set(nodes))) != 1)
except KeyError:
    pass
