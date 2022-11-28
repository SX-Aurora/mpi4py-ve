from mpi4pyve import MPI
import numpy as np
import nlcpy as vp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dev', type=str, required=True, choices=['vh', 've'],
                    help='Execution device')
parser.add_argument('-dtype', type=str, required=True, choices=['float', 'double'],
                    help='Execution data type')
parser.add_argument('-m', type=int, required=False, default=10,
                    help='Number of rows of matrix A')
parser.add_argument('-n', type=int, required=False, default=10,
                    help='Number of cols of matrix A and number of vector x')
parser.add_argument('-iter', type=int, required=False, default=10000,
                    help='Number of iterations for gemv')
args = parser.parse_args()

# set module
if args.dev == 'vh':
    dev = np
elif args.dev == 've':
    dev = vp
else:
    raise ValueError

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0: print(vars(args))

m = args.m
n = args.n
if args.dtype == 'float':
    dtype = 'f4'
    mpi_dtype = MPI.FLOAT
elif args.dtype == 'double':
    dtype = 'f8'
    mpi_dtype = MPI.DOUBLE
else:
    raise ValueError

# estimate send/recv count and displacement
A_count = [0 for _ in range(size)]
A_displ = [0 for _ in range(size)]
y_count = [0 for _ in range(size)]
y_displ = [0 for _ in range(size)]
for i in range(size):
    m_s = m * i // size
    m_e = m * (i + 1) // size
    if i == rank:
        m_d = m_e - m_s
    A_count[i] = (m_e - m_s) * n
    A_displ[i] = m_s * n
    y_count[i] = (m_e - m_s)
    y_displ[i] = m_s

# create matrix A and vector x
rng = dev.random.default_rng()
if rank == 0:
    A = rng.random((m, n), dtype=dtype)
    x = rng.random(n, dtype=dtype)
else:
    A = None
    x = dev.zeros(n, dtype=dtype)
A_local = dev.empty((m_d, n), dtype=dtype)
comm.Scatterv([A, A_count, A_displ, mpi_dtype], [A_local, A_count[rank], mpi_dtype], root=0)  # divide matrix A into each process.
comm.Bcast(x, root=0)  # all processes share same vector x.

# execute gemv
if dev.__name__ == 'nlcpy':
    dev.request.flush()
comm.Barrier()
t0 = MPI.Wtime()
for _ in range(args.iter):
    y_local = A_local @ x  # local gemv
if dev.__name__ == 'nlcpy':
    dev.request.flush()
comm.Barrier()
t1 = MPI.Wtime()

# gather local vector y into root process
if rank == 0:
    y = dev.empty(m, dtype=dtype)
else:
    y = None
comm.Gatherv([y_local, y_count[rank], mpi_dtype], [y, y_count, y_displ, mpi_dtype], root=0)

if rank == 0:
    print("elapsed:", t1 - t0, "[sec]")

# result check
if rank == 0:
    res = dev.all((y - A @ x) / y < 1e-4)
    print("Result {}".format("success" if res else "failed"))
