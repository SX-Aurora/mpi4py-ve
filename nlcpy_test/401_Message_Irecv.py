from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("rank = ",rank)

    x = vp.array([1,2,3], dtype=int)
    print("x       = ",x)
    print("type(x) = ",type(x))
    comm.Send([x, MPI.INT], dest=1)

elif rank == 1:
    print("rank = ",rank)
    y = vp.empty(3, dtype=int)
    m = comm.Mprobe(0, 0)
    req = m.Irecv(y)
    req.Wait()
    print("y       = ",y)
    print("type(y) = ",type(y)) 

    import sys
    try:
        y
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)

comm.barrier()

if rank == 0:
    print("rank = ",rank)

    x = vp.array([1,2,3], dtype=int)
    print("x       = ",x)
    print("type(x) = ",type(x))
    comm.Send([x, MPI.INT], dest=1)

elif rank == 1:
    print("rank = ",rank)
    y = vp.empty(3, dtype=int)
    m = comm.Mprobe(0, 0)
    req = m.Irecv([y, MPI.INT])
    req.Wait()
    print("y       = ",y)
    print("type(y) = ",type(y))

    import sys
    try:
        y
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)

