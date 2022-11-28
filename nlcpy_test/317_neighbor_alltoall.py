from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD.Create_cart((MPI.COMM_WORLD.Get_size(),))
size = comm.Get_size()
rank = comm.Get_rank()

print("rank = ",rank)

x = vp.array([[(rank+1)**2 ,rank+1],[(rank+1)**3 ,-(rank+1)]], dtype=int)
print("x       = ",x)
print("type(x) = ",type(x))

y = comm.neighbor_alltoall(x)

print("neighbor alltoall done")

print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y[0]) = ",type(y[0]))
print("type(y[1]) = ",type(y[1])) 
import sys
try:
    y
    if not isinstance(y[0], vp.core.core.ndarray) and\
       not isinstance(y[1], vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
