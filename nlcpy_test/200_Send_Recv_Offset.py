from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("rank = ",rank)

    #x = vp.arange(10)[::2]
    x = vp.arange(10)[2:]
    print("x       = ",x)
    print("type(x) = ",type(x))
    comm.Send([x, MPI.INT], dest=1)

elif rank == 1:
    print("rank = ",rank)
    
    y = vp.empty(10-2, dtype=int)
    comm.Recv([y, MPI.INT])
    print("y       = ",y)
    print("type(y) = ",type(y)) 

    import sys
    try:
        y
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)
