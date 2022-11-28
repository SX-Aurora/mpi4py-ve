from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

print("rank = ",rank)

datatype = MPI.INT

x = vp.array([(rank+1)**2 , rank], dtype=int)
y = vp.zeros(2, dtype=int)

size1 = datatype.Pack_size(len(x), comm)
size2 = datatype.Pack_size(len(y), comm)
tmpbuf = vp.zeros(size1 + size2 + 1, dtype=int)

print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))

datatype.Pack(x, tmpbuf, 0, comm)
datatype.Unpack(tmpbuf, 0, y, comm)

print("Pack-Unpack done")

print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))

import sys
try:
    pass
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
