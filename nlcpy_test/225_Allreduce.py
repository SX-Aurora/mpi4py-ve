from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

print("rank = ",rank)

x = vp.array([(rank+1)**2 , rank], dtype=int)
y = vp.zeros(2, dtype=int)

print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))

comm.Allreduce(x, y)

print("Allreduce done")
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
