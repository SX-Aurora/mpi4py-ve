from mpi4pyve import MPI
import numpy as np
import nlcpy as vp
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("rank = ",rank)

    x = vp.array([1,2,3], dtype=int)
    comm.Send([x, MPI.INT], dest=1)

    a = vp.empty(3, dtype=int)
    comm.Recv([a, MPI.INT])
    #a = comm.Sendrecv([x, MPI.INT], 1)
    print("a       = ",a)
    print("type(a) = ",type(a))
    try:
        a
        if not isinstance(a, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)

elif rank == 1:
    print("rank = ",rank)
    
    y = vp.array([4,5,6], dtype=int)
    z = vp.empty(3, dtype=int)
    comm.Sendrecv([y, MPI.INT], 0, recvbuf=[z, MPI.INT])
    print("z       = ",z)
    print("type(z) = ",type(z))

    try:
        z
        if not isinstance(z, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)
