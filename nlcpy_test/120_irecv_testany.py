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
y = MPI.Request.testany([req])
print("y       = ",y)
print("type(y[2]) = ",type(y[2])) 

import sys
try:
    y[2]
    if not isinstance(y[2], vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
