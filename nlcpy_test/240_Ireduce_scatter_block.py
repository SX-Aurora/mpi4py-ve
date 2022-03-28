from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

print("rank = ",rank)

#x = vp.arange(size, dtype=int) * (rank + 1)
x = vp.array([[rank+1 for i in range(size)] for j in range(size) ], dtype=int)
y = vp.empty(3, dtype=int)

print("x       = ",x)
print("type(x) = ",type(x))

comm.Ireduce_scatter_block(x, y).Wait()

print("Ireduce_scatter_block done")
print("y       = ",y)
print("type(y) = ",type(y)) 
import sys
try:
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
