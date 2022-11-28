from mpi4pyve import MPI
import numpy as np
import nlcpy as vp
from utils_io import get_fh

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

fh = get_fh()
fh.Set_size(0)
fh.Set_view(rank*12, MPI.INT)

x = vp.array([1,2,3], dtype=int)
y = vp.zeros(3, dtype=int)

print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))

fh.Seek(rank*12, MPI.SEEK_SET)
req = fh.Iwrite_all(x)
req.Wait()
fh.Sync()
comm.Barrier()
fh.Sync()
fh.Seek(rank*12, MPI.SEEK_SET)
req = fh.Iread_all(y)
req.Wait()
comm.Barrier()

print("Iwrite_all-Iread_all done")

print("x       = ",x)
print("type(x) = ",type(x))
print("y       = ",y)
print("type(y) = ",type(y))

if fh:
    fh.Close()
comm.Barrier()

import sys
try:
    y
    if not isinstance(y, vp.core.core.ndarray):
        print("NG : ", __file__, file=sys.stderr)
except NameError:
    print("Failure test case : ", __file__, file=sys.stderr)
