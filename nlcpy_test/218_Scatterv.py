from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0
print("rank = ",rank)

#if rank == root:
#    x = [vp.arange(size + i) for i in range(size)]
#else:
#    x = None
#y = vp.empty(5, dtype=int)

if rank == root:
    x = vp.asarray([vp.arange(size, dtype=int) * (rank + 1),
         vp.arange(size, dtype=int) * (rank + 2),
         vp.arange(size, dtype=int) * (rank + 3)], dtype=int)
else:
    x = None

y = vp.empty(3, dtype=int)



print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))

x = comm.Scatterv(x, y, root=root)

print("Scatterv done")
print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))


#if rank == root:
#    print("type(x[0]) = ",type(x[0]))
#    print("type(x[1]) = ",type(x[1]))
#    print("type(x[2]) = ",type(x[2])) 
import sys
try:
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
