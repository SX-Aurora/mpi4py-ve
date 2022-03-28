from mpi4pyve import MPI
import numpy as np
import nlcpy as vp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

print("rank = ",rank)

datatype = MPI.INT
EXT32 = 'external32'

x = vp.array([(rank+1)**2 , rank], dtype=int)
y = vp.empty(2, dtype=int)

size1 = datatype.Pack_external_size(EXT32, x.size)
size2 = datatype.Pack_external_size(EXT32, len(y))
tmpbuf = vp.empty(size1 + size2 + 1, dtype=int)

print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))

datatype.Pack_external(EXT32, x, tmpbuf, 0)
datatype.Unpack_external(EXT32, tmpbuf, 0, y)

print("Pack_external-Unpack_external done")

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
