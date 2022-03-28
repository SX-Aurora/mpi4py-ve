from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


x = vp.arange(size, dtype=int) * (rank + 1)

#if rank == 0:
#    x = vp.array([1,2,3], dtype=int)
#else:
#    x = vp.empty(3, dtype=int)

print("rank = ",rank)

print("x       = ",x)
print("type(x) = ",type(x)) 

#comm.Ibcast(x, root=0).Wait()
comm.Ibcast([x, MPI.INT], root=0).Wait()

print("Ibcast done")

print("x       = ",x)
print("type(x) = ",type(x)) 
import sys
try:
    x
    if not isinstance(x, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
