from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("rank = ",rank)

    x = vp.array([1,2,3])
    print("x       = ",x)
    print("type(x) = ",type(x))
    comm.send(x, dest=1)

elif rank == 1:
    print("rank = ",rank)

    req = comm.irecv()
    y = MPI.Request.waitall([req])
    print("y       = ",y)
    print("type(y[0]) = ",type(y[0])) 

    import sys
    try:
        y[0]
        if not isinstance(y[0], vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)
