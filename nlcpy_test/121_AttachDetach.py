from mpi4pyve import MPI
import numpy as np
import nlcpy as vp
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("rank = ",rank)

    x = vp.arange(200000, dtype=int)
    print(x.__ve_array_interface__) 
    MPI.Attach_buffer(x)
    MPI.Detach_buffer()

