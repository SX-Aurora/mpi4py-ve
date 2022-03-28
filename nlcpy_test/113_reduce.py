from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

print("rank = ",rank)

x = vp.array([(rank+1)**2 , rank])
print("x       = ",x)
print("type(x) = ",type(x))

x = comm.reduce(x, root=root)

print("reduce done")
print("x       = ",x)
print("type(x) = ",type(x)) 

if rank==root:
    import sys
    try:
        x
        if not isinstance(x, vp.core.core.ndarray):
            print("NG : ", __file__, file=sys.stderr)
    except NameError:
        print("Failure test case : ", __file__, file=sys.stderr)
