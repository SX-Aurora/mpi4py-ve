from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("rank = ",rank)

x = vp.arange(size**2, dtype=int).reshape(size, size) * (rank + 1)
y = vp.zeros((size, size), dtype=int)
print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))


comm.Alltoallv(x, y)

print("Alltoallv done")
print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y)) 
import sys
try:
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
