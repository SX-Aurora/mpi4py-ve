from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

x = vp.array([1,2,3])
print("x       = ",x)
print("type(x) = ",type(x))
comm.send(x, dest=rank)

req = comm.irecv(source=rank)
y = MPI.Request.testall([req])
print("y       = ",y)
print("type(y[1][0]) = ",type(y[1][0])) 

import sys
try:
    y[1][0]
    if not isinstance(y[1][0], vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
