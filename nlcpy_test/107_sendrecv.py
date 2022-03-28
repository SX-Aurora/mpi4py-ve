from mpi4pyve import MPI
import numpy as np
import nlcpy as vp
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("rank = ",rank)

    x = vp.array([1,2,3])
    comm.send(x, dest=1)

    y = comm.recv()
    print("y       = ",y)
    print("type(y) = ",type(y))

    try:
        y
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)


elif rank == 1:
    z = vp.array([4,5,6])

    w = comm.sendrecv(z, dest=0)
    print("w       = ",w)
    print("type(w) = ",type(w)) 

    try:
        w
        if not isinstance(w, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)

