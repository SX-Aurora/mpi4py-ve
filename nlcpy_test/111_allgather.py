from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("rank = ",rank)

x = vp.array([(rank+1)**2 , rank])
print("x       = ",x)
print("type(x) = ",type(x))

x = comm.allgather(x)

print("allgather done")
print("x       = ",x)

for i in range(rank):
    print("type(x[{}]) = ".format(i),type(x[i]))
import sys
try:
    for y in x:
        if not isinstance(y, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
