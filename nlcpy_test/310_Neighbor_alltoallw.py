from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

dim = 3
comm = MPI.COMM_WORLD.Create_cart((MPI.COMM_WORLD.Get_size(),))
size = comm.Get_size()
rank = comm.Get_rank()

print("rank = ",rank)

x = vp.array([[(rank+1)**(rank+1), rank + 1],
              [(rank + 1)*2, ((rank+1)**(rank+1))*2]], dtype=int)
send=[x, None, None, [MPI.INT for i in range(0, dim-1)]]

y = vp.zeros((2,2), dtype=int)
recv=[y, [MPI.INT for i in range(0, dim-1)]]
print("x       = ",x)
print("type(x) = ",type(x))

comm.Neighbor_alltoallw(send, recv)

print("Neighbor alltoallw done")

print("y       = ",y)
print("type(y) = ",type(y)) 
import sys
try:
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
