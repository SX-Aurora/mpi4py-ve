from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("rank = ",rank)

x = vp.array([[(rank+1)**(rank+1), rank + 1],
              [(rank + 1)*2, ((rank+1)**(rank+1))*2]], dtype=int)
send=[x, None, None, [MPI.INT for i in range(0, size)]]

y = vp.zeros((1,), dtype=int)
recv=[y, [MPI.INT for i in range(0, size)]]

print("x       = ",x)
print("type(x) = ",type(x))

comm.Alltoallw(send, recv)

print("Alltoallw done")
print("y       = ",y)
print("type(y) = ",type(y)) 
import sys
try:
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
