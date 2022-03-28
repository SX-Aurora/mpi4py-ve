from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("rank = ",rank)

a = vp.arange(size**2, dtype=int).reshape(size, size) * (rank + 1)
print("a       = ",a)

x = comm.alltoall(a)

print("allgather done")
print("x       = ",x)

print("type(x[0]) = ",type(x[0]))
print("type(x[1]) = ",type(x[1]))
print("type(x[2]) = ",type(x[2])) 
import sys
try:
    for y in x:
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
