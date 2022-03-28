from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

dim = 3
comm = MPI.COMM_WORLD.Create_cart((dim,))
size = comm.Get_size()
rank = comm.Get_rank()

print("rank = ",rank)

x = vp.array([[(rank+1)**(rank+1), rank + 1],
              [(rank + 1)*2, ((rank+1)**(rank+1))*2]], dtype=int)

y = vp.zeros((2,2), dtype=int)
print("x       = ",x)
print("type(x) = ",type(x))

comm.Ineighbor_alltoall(x, y).Wait()

print("Ineighbor alltoall done")

print("y       = ",y)
print("type(y) = ",type(y)) 
import sys
try:
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
