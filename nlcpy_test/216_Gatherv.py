from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

x = vp.array([(rank+1)**2, rank], dtype=int)
if rank == root:
    y = vp.empty((size, 2), dtype=int)
else:
    y = None

print("rank = ",rank)

print("x       = ",x)
print("type(x) = ",type(x))

comm.Gatherv(x, y, root=root)

print("Gatherv done")

print("y       = ",y)
print("type(y) = ",type(y)) 
if rank==root:
    import sys
    try:
        y
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)
