from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("rank = ",rank)

    x = vp.array([1,2,3], dtype=int)
    #comm.Send(x, dest=1)
    comm.Send([x, MPI.INT], dest=1)

    a = vp.empty(3, dtype=int)
    #comm.Recv(a)
    comm.Recv([a, MPI.INT])
    print("a       = ",a)
    print("type(a) = ",type(a))

elif rank == 1:
    print("rank = ",rank)
    
    y = vp.array([4,5,6], dtype=int)
    #comm.Sendrecv_replace(y, 0)
    comm.Sendrecv_replace([y, MPI.INT], 0)
    print("y       = ",y)
    print("type(y) = ",type(y)) 

    import sys
    try:
        y
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)
